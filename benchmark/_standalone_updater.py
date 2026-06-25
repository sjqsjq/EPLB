"""
Standalone copy of update_expert_weights_single_layer and create_temp_buffers
from sglang.srt.eplb.expert_location_updater, without the heavy sglang imports.
"""
from typing import List, Optional, Tuple
import torch
import torch.distributed
from torch.distributed import P2POp


def create_temp_buffers(sample_tensors):
    return [torch.empty_like(tensor) for tensor in sample_tensors]


def _deduplicate_ordered(lst):
    seen = set()
    result = []
    for x in lst:
        if x not in seen:
            seen.add(x)
            result.append(x)
    return result


class _ChunkUtils:
    def __init__(self, *, chunk_values: List, element_values: List):
        self.chunk_values = chunk_values
        self.element_values = element_values

    def chunk_value_from_element_value(self, element_value):
        chunk_index = self._chunk_index_from_element_index(
            num_elements=len(self.element_values),
            num_chunks=len(self.chunk_values),
            element_index=self.element_values.index(element_value),
        )
        return self.chunk_values[chunk_index]

    def element_values_from_chunk_value(self, chunk_value) -> List:
        if len(self.element_values) == 0:
            return []
        element_slice = self._element_slice_from_chunk_index(
            num_elements=len(self.element_values),
            num_chunks=len(self.chunk_values),
            chunk_index=self.chunk_values.index(chunk_value),
        )
        return self.element_values[element_slice]

    @staticmethod
    def _chunk_index_from_element_index(num_elements, num_chunks, element_index):
        short_chunk_size, num_long_chunks = divmod(num_elements, num_chunks)
        num_elements_for_long_chunks = num_long_chunks * (short_chunk_size + 1)
        if element_index < num_elements_for_long_chunks:
            return element_index // (short_chunk_size + 1)
        else:
            return num_long_chunks + (element_index - num_elements_for_long_chunks) // short_chunk_size

    @staticmethod
    def _element_slice_from_chunk_index(num_elements, num_chunks, chunk_index):
        short_chunk_size, num_long_chunks = divmod(num_elements, num_chunks)
        if chunk_index < num_long_chunks:
            start = chunk_index * (short_chunk_size + 1)
            end = start + short_chunk_size + 1
        else:
            start = num_long_chunks * (short_chunk_size + 1) + (chunk_index - num_long_chunks) * short_chunk_size
            end = start + short_chunk_size
        return slice(start, end)


def update_expert_weights_single_layer(
    routed_experts_weights: List[torch.Tensor],
    temp_buffers: List[torch.Tensor],
    old_physical_to_logical_map: List[int],
    new_physical_to_logical_map: List[int],
    num_local_physical_experts: int,
    num_gpu_per_node: int,
    rank: int,
    world_size: Optional[int] = None,
):
    assert all(
        tensor.shape[0] == num_local_physical_experts
        for tensor in routed_experts_weights
    )

    num_physical_experts = len(old_physical_to_logical_map)
    num_tensors = len(routed_experts_weights)
    self_node_id = rank // num_gpu_per_node

    local_expert_location_range = (
        rank * num_local_physical_experts,
        (rank + 1) * num_local_physical_experts,
    )

    def _entrypoint():
        p2p_op_infos: List[Tuple[int, List[P2POp]]] = []
        buffer2weight_copy_infos: List[Tuple[int, int]] = []

        _handle_recv(buffer2weight_copy_infos, p2p_op_infos)
        _create_isend_ops(p2p_op_infos)
        _execute_p2p_ops(p2p_op_infos)
        _execute_buffer2weight_copies(buffer2weight_copy_infos)

    def _handle_recv(buffer2weight_copy_infos, p2p_op_infos):
        for dst_expert_location in range(*local_expert_location_range):
            _handle_recv_of_dst(dst_expert_location, buffer2weight_copy_infos, p2p_op_infos)

    def _handle_recv_of_dst(dst_expert_location, buffer2weight_copy_infos, p2p_op_infos):
        logical_expert_id = new_physical_to_logical_map[dst_expert_location]

        # case 1: unchanged
        if old_physical_to_logical_map[dst_expert_location] == logical_expert_id:
            return

        # case 2: same-gpu
        for src_expert_location in range(*local_expert_location_range):
            if old_physical_to_logical_map[src_expert_location] == logical_expert_id:
                for i in range(num_tensors):
                    _get_tensor(temp_buffers, i, dst_expert_location).copy_(
                        _get_tensor(routed_experts_weights, i, src_expert_location)
                    )
                buffer2weight_copy_infos.append((dst_expert_location, dst_expert_location))
                return

        # case 3: free-rider
        for src_expert_location in range(rank * num_local_physical_experts, dst_expert_location):
            if new_physical_to_logical_map[src_expert_location] == logical_expert_id:
                buffer2weight_copy_infos.append((src_expert_location, dst_expert_location))
                return

        same_node_mapping, cross_node_mapping, need_comm_self_node_dst_ranks = (
            _compute_comm_info(logical_expert_id=logical_expert_id)
        )

        # case 4: same-node
        if rank in need_comm_self_node_dst_ranks:
            chosen_src_rank = same_node_mapping.chunk_value_from_element_value(element_value=rank)
            _create_p2p_recv(buffer2weight_copy_infos, p2p_op_infos,
                           src_rank=chosen_src_rank, logical_expert_id=logical_expert_id,
                           dst_expert_location=dst_expert_location)
            return

        # case 5: cross-node
        chosen_src_rank = cross_node_mapping.chunk_value_from_element_value(element_value=rank)
        _create_p2p_recv(buffer2weight_copy_infos, p2p_op_infos,
                        src_rank=chosen_src_rank, logical_expert_id=logical_expert_id,
                        dst_expert_location=dst_expert_location)

    def _create_p2p_recv(buffer2weight_copy_infos, p2p_op_infos, *, logical_expert_id, src_rank, dst_expert_location):
        p2p_op_infos.append((
            logical_expert_id,
            [P2POp(op=torch.distributed.irecv, tensor=_get_tensor(temp_buffers, i, dst_expert_location), peer=src_rank)
             for i in range(num_tensors)]
        ))
        buffer2weight_copy_infos.append((dst_expert_location, dst_expert_location))

    def _create_isend_ops(p2p_op_infos):
        handled = set()
        for src_expert_location in range(*local_expert_location_range):
            logical_expert_id = old_physical_to_logical_map[src_expert_location]
            if logical_expert_id in handled:
                continue
            handled.add(logical_expert_id)
            _create_isend_ops_of(logical_expert_id, src_expert_location, p2p_op_infos)

    def _create_isend_ops_of(logical_expert_id, src_expert_location, p2p_op_infos):
        same_node_mapping, cross_node_mapping, _ = _compute_comm_info(logical_expert_id=logical_expert_id)
        same_node_dst_ranks = same_node_mapping.element_values_from_chunk_value(chunk_value=rank)
        cross_node_dst_ranks = cross_node_mapping.element_values_from_chunk_value(chunk_value=rank)
        all_dst_ranks = same_node_dst_ranks + cross_node_dst_ranks

        p2p_op_infos.append((
            logical_expert_id,
            [P2POp(op=torch.distributed.isend,
                   tensor=_get_tensor(routed_experts_weights, i, src_expert_location),
                   peer=dst_rank)
             for dst_rank in all_dst_ranks for i in range(num_tensors)]
        ))

    def _compute_comm_info(logical_expert_id):
        all_src_ranks = _deduplicate_ordered([
            x // num_local_physical_experts for x in range(num_physical_experts)
            if old_physical_to_logical_map[x] == logical_expert_id
        ])
        all_src_nodes = [x // num_gpu_per_node for x in all_src_ranks]
        self_node_src_ranks = [x for x in all_src_ranks if x // num_gpu_per_node == self_node_id]

        need_comm_dst_ranks = _deduplicate_ordered([
            x // num_local_physical_experts for x in range(num_physical_experts)
            if new_physical_to_logical_map[x] == logical_expert_id
            and x // num_local_physical_experts not in all_src_ranks
        ])
        need_comm_self_node_dst_ranks = (
            [x for x in need_comm_dst_ranks if x // num_gpu_per_node == self_node_id]
            if len(self_node_src_ranks) > 0 else []
        )
        need_comm_cross_node_dst_ranks = [
            x for x in need_comm_dst_ranks
            if (x // num_gpu_per_node) not in all_src_nodes
        ]

        same_node_mapping = _ChunkUtils(chunk_values=self_node_src_ranks, element_values=need_comm_self_node_dst_ranks)
        cross_node_mapping = _ChunkUtils(chunk_values=all_src_ranks, element_values=need_comm_cross_node_dst_ranks)

        return same_node_mapping, cross_node_mapping, need_comm_self_node_dst_ranks

    def _execute_p2p_ops(p2p_op_infos):
        sorted_infos = sorted(p2p_op_infos, key=lambda info: info[0])
        p2p_ops = [op for _, ops in sorted_infos for op in ops]
        if len(p2p_ops) == 0:
            return
        reqs = torch.distributed.batch_isend_irecv(p2p_ops)
        for req in reqs:
            req.wait()

    def _execute_buffer2weight_copies(buffer2weight_copy_infos):
        for temp_loc, weight_loc in buffer2weight_copy_infos:
            for i in range(num_tensors):
                _get_tensor(routed_experts_weights, i, weight_loc).copy_(
                    _get_tensor(temp_buffers, i, temp_loc))

    def _get_tensor(tensors, tensor_index, expert_location):
        return tensors[tensor_index][expert_location % num_local_physical_experts]

    _entrypoint()
