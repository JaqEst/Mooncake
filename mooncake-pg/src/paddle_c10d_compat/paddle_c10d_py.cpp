#include <paddle_c10d_compat/pybind.h>

#include <pybind11/gil.h>
#include <pybind11/stl.h>
#include <pybind11/chrono.h>
#include <torch/python.h>

#include <paddle_c10d_compat/backend.h>
#include <paddle_c10d_compat/store.h>
#include <paddle_c10d_compat/types.h>
#include <paddle_c10d_compat/work.h>

namespace py = pybind11;

namespace mooncake::paddle_c10d_compat {

void bindC10dCompat(py::module_& m) {
    py::enum_<c10d::OpType>(m, "OpType")
        .value("BROADCAST", c10d::OpType::BROADCAST)
        .value("ALLREDUCE", c10d::OpType::ALLREDUCE)
        .value("REDUCE", c10d::OpType::REDUCE)
        .value("ALLGATHER", c10d::OpType::ALLGATHER)
        .value("GATHER", c10d::OpType::GATHER)
        .value("SCATTER", c10d::OpType::SCATTER)
        .value("REDUCE_SCATTER", c10d::OpType::REDUCE_SCATTER)
        .value("ALLTOALL", c10d::OpType::ALLTOALL)
        .value("SEND", c10d::OpType::SEND)
        .value("RECV", c10d::OpType::RECV)
        .value("BARRIER", c10d::OpType::BARRIER)
        .export_values();

    py::class_<c10d::ReduceOp> reduce_op(m, "ReduceOp");
    reduce_op.def(py::init<c10d::ReduceOp::RedOpType>())
        .def_readwrite("op", &c10d::ReduceOp::op_);

    py::enum_<c10d::ReduceOp::RedOpType>(reduce_op, "RedOpType")
        .value("SUM", c10d::ReduceOp::RedOpType::SUM)
        .value("AVG", c10d::ReduceOp::RedOpType::AVG)
        .value("PRODUCT", c10d::ReduceOp::RedOpType::PRODUCT)
        .value("MIN", c10d::ReduceOp::RedOpType::MIN)
        .value("MAX", c10d::ReduceOp::RedOpType::MAX)
        .value("BAND", c10d::ReduceOp::RedOpType::BAND)
        .value("BOR", c10d::ReduceOp::RedOpType::BOR)
        .value("BXOR", c10d::ReduceOp::RedOpType::BXOR)
        .export_values();

    py::implicitly_convertible<c10d::ReduceOp::RedOpType, c10d::ReduceOp>();

    py::class_<c10d::BroadcastOptions>(m, "BroadcastOptions")
        .def(py::init<>())
        .def_readwrite("root_rank", &c10d::BroadcastOptions::rootRank)
        .def_readwrite("root_tensor", &c10d::BroadcastOptions::rootTensor)
        .def_readwrite("timeout", &c10d::BroadcastOptions::timeout)
        .def_readwrite("async_op", &c10d::BroadcastOptions::asyncOp);

    py::class_<c10d::AllreduceOptions>(m, "AllreduceOptions")
        .def(py::init<>())
        .def_readwrite("reduce_op", &c10d::AllreduceOptions::reduceOp)
        .def_readwrite("timeout", &c10d::AllreduceOptions::timeout)
        .def_readwrite("async_op", &c10d::AllreduceOptions::asyncOp);

    py::class_<c10d::AllgatherOptions>(m, "AllgatherOptions")
        .def(py::init<>())
        .def_readwrite("timeout", &c10d::AllgatherOptions::timeout)
        .def_readwrite("async_op", &c10d::AllgatherOptions::asyncOp);

    py::class_<c10d::ReduceScatterOptions>(m, "ReduceScatterOptions")
        .def(py::init<>())
        .def_readwrite("reduce_op", &c10d::ReduceScatterOptions::reduceOp)
        .def_readwrite("timeout", &c10d::ReduceScatterOptions::timeout)
        .def_readwrite("async_op", &c10d::ReduceScatterOptions::asyncOp);

    py::class_<c10d::AllToAllOptions>(m, "AllToAllOptions")
        .def(py::init<>())
        .def_readwrite("timeout", &c10d::AllToAllOptions::timeout)
        .def_readwrite("async_op", &c10d::AllToAllOptions::asyncOp);

    py::class_<c10d::BarrierOptions>(m, "BarrierOptions")
        .def(py::init<>())
        .def_readwrite("device_ids", &c10d::BarrierOptions::device_ids)
        .def_readwrite("timeout", &c10d::BarrierOptions::timeout)
        .def_readwrite("device", &c10d::BarrierOptions::device)
        .def_readwrite("async_op", &c10d::BarrierOptions::asyncOp);

    py::class_<c10d::ReduceOptions>(m, "ReduceOptions")
        .def(py::init<>())
        .def_readwrite("reduce_op", &c10d::ReduceOptions::reduceOp)
        .def_readwrite("root_rank", &c10d::ReduceOptions::rootRank)
        .def_readwrite("root_tensor", &c10d::ReduceOptions::rootTensor)
        .def_readwrite("timeout", &c10d::ReduceOptions::timeout)
        .def_readwrite("async_op", &c10d::ReduceOptions::asyncOp);

    py::class_<c10d::GatherOptions>(m, "GatherOptions")
        .def(py::init<>())
        .def_readwrite("root_rank", &c10d::GatherOptions::rootRank)
        .def_readwrite("timeout", &c10d::GatherOptions::timeout)
        .def_readwrite("async_op", &c10d::GatherOptions::asyncOp);

    py::class_<c10d::ScatterOptions>(m, "ScatterOptions")
        .def(py::init<>())
        .def_readwrite("root_rank", &c10d::ScatterOptions::rootRank)
        .def_readwrite("timeout", &c10d::ScatterOptions::timeout)
        .def_readwrite("async_op", &c10d::ScatterOptions::asyncOp);

    py::module_::import("paddle.base.core");
    py::class_<c10d::PrefixStore, c10d::Store,
               c10::intrusive_ptr<c10d::PrefixStore>>(m, "PrefixStore")
        .def(py::init<std::string, c10::intrusive_ptr<c10d::Store>>(),
              py::arg("prefix"), py::arg("store"));

    py::class_<c10d::DistributedBackendOptions>(m, "DistributedBackendOptions")
        .def(py::init<>())
        .def_readwrite("store", &c10d::DistributedBackendOptions::store)
        .def_readwrite("group_rank", &c10d::DistributedBackendOptions::group_rank)
        .def_readwrite("group_size", &c10d::DistributedBackendOptions::group_size)
        .def_readwrite("timeout", &c10d::DistributedBackendOptions::timeout)
        .def_readwrite("group_id", &c10d::DistributedBackendOptions::group_id)
        .def_readwrite("global_ranks_in_group", &c10d::DistributedBackendOptions::global_ranks_in_group);

    py::class_<c10d::Work, c10::intrusive_ptr<c10d::Work>>(m, "Work")
        .def("is_completed", &c10d::Work::isCompleted)
        .def("is_success", &c10d::Work::isSuccess)
        .def("wait", &c10d::Work::wait,
             py::arg("timeout") = kNoTimeout,
             py::call_guard<py::gil_scoped_release>())
        .def("synchronize", &c10d::Work::synchronize)
        .def("result", &c10d::Work::result);

    py::class_<c10d::Backend, c10::intrusive_ptr<c10d::Backend>>(m, "Backend")
        .def("rank", &c10d::Backend::getRank)
        .def("size", &c10d::Backend::getSize)
        .def("name", &c10d::Backend::getBackendName)
        .def("send", &c10d::Backend::send,
             py::arg("tensors"), py::arg("dst_rank"), py::arg("tag"),
             py::call_guard<py::gil_scoped_release>())
        .def("recv", &c10d::Backend::recv,
             py::arg("tensors"), py::arg("src_rank"), py::arg("tag"),
             py::call_guard<py::gil_scoped_release>())
        .def("broadcast", &c10d::Backend::broadcast,
             py::arg("tensors"), py::arg("opts") = c10d::BroadcastOptions(),
             py::call_guard<py::gil_scoped_release>())
        .def("allreduce", &c10d::Backend::allreduce,
             py::arg("tensors"), py::arg("opts") = c10d::AllreduceOptions(),
             py::call_guard<py::gil_scoped_release>())
        .def("allgather", &c10d::Backend::allgather,
             py::arg("output_tensors"), py::arg("input_tensors"),
             py::arg("opts") = c10d::AllgatherOptions(),
             py::call_guard<py::gil_scoped_release>())
        .def("_allgather_base", &c10d::Backend::_allgather_base,
             py::arg("output_buffer"), py::arg("input_buffer"),
             py::arg("opts") = c10d::AllgatherOptions(),
             py::call_guard<py::gil_scoped_release>())
        .def("_reduce_scatter_base", &c10d::Backend::_reduce_scatter_base,
             py::arg("output_buffer"), py::arg("input_buffer"),
             py::arg("opts") = c10d::ReduceScatterOptions(),
             py::call_guard<py::gil_scoped_release>())
        .def("alltoall", &c10d::Backend::alltoall,
             py::arg("output_tensors"), py::arg("input_tensors"),
             py::arg("opts") = c10d::AllToAllOptions(),
             py::call_guard<py::gil_scoped_release>())
        .def("barrier", &c10d::Backend::barrier,
             py::arg("opts") = c10d::BarrierOptions(),
             py::call_guard<py::gil_scoped_release>())
        .def("reduce", &c10d::Backend::reduce,
             py::arg("tensors"), py::arg("opts") = c10d::ReduceOptions(),
             py::call_guard<py::gil_scoped_release>())
        .def("gather", &c10d::Backend::gather,
             py::arg("output_tensors"), py::arg("input_tensors"),
             py::arg("opts") = c10d::GatherOptions(),
             py::call_guard<py::gil_scoped_release>())
        .def("scatter", &c10d::Backend::scatter,
             py::arg("output_tensors"), py::arg("input_tensors"),
             py::arg("opts") = c10d::ScatterOptions(),
             py::call_guard<py::gil_scoped_release>())
        .def("shutdown", &c10d::Backend::shutdown);
}

} // namespace mooncake::paddle_c10d_compat
