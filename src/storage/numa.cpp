// OS allocation boundary only: no attention/ANN kernels or GPU dependency.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <numaif.h>
#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <system_error>
#include <vector>

namespace py = pybind11;

static void syscall_error(const char* operation) {
    throw std::system_error(errno, std::generic_category(), operation);
}

class Region : public std::enable_shared_from_this<Region> {
public:
    Region(std::size_t requested, int node) : requested_(requested), node_(node) {
        const long page = sysconf(_SC_PAGESIZE);
        if (page <= 0 || requested == 0 || node < 0 || node > 65535 ||
            requested > static_cast<std::size_t>(std::numeric_limits<py::ssize_t>::max()) - page) {
            throw std::invalid_argument("invalid allocation size or NUMA node");
        }
        page_ = static_cast<std::size_t>(page);
        mapped_ = ((requested + page_ - 1) / page_) * page_;
        data_ = mmap(nullptr, mapped_, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (data_ == MAP_FAILED) syscall_error("mmap");
        try {
            constexpr std::size_t bits = sizeof(unsigned long) * 8;
            std::vector<unsigned long> mask(static_cast<std::size_t>(node) / bits + 1, 0);
            mask[static_cast<std::size_t>(node) / bits] |= 1UL << (static_cast<std::size_t>(node) % bits);
            if (mbind(data_, mapped_, MPOL_BIND, mask.data(), static_cast<unsigned long>(node) + 1,
                      MPOL_MF_STRICT) != 0) syscall_error("mbind(MPOL_BIND)");
            // Small base pages keep the first infrastructure path and accounting explicit.
            if (madvise(data_, mapped_, MADV_NOHUGEPAGE) != 0) syscall_error("madvise(MADV_NOHUGEPAGE)");
            std::memset(data_, 0, mapped_);
            const auto observed = residency();
            if (observed.size() != 1 || observed.begin()->first != node_ ||
                observed.begin()->second != mapped_ / page_) {
                throw std::runtime_error("page residency differs from requested node; allocation rejected");
            }
        } catch (...) {
            munmap(data_, mapped_);
            data_ = MAP_FAILED;
            throw;
        }
    }

    ~Region() { if (data_ != MAP_FAILED) munmap(data_, mapped_); }
    Region(const Region&) = delete;
    Region& operator=(const Region&) = delete;

    std::map<int, std::size_t> residency() const {
        std::map<int, std::size_t> counts;
        constexpr std::size_t batch = 4096;
        const std::size_t pages = mapped_ / page_;
        for (std::size_t first = 0; first < pages; first += batch) {
            const std::size_t count = std::min(batch, pages - first);
            std::vector<void*> addresses(count);
            std::vector<int> status(count, -1);
            for (std::size_t i = 0; i < count; ++i) {
                addresses[i] = static_cast<char*>(data_) + (first + i) * page_;
            }
            // Query this process. Never request migration or inspect other jobs.
            if (move_pages(0, count, addresses.data(), nullptr, status.data(), 0) < 0) {
                syscall_error("move_pages(residency query)");
            }
            for (int observed : status) {
                if (observed < 0) {
                    throw std::system_error(-observed, std::generic_category(), "page residency query");
                }
                ++counts[observed];
            }
        }
        return counts;
    }

    py::array bytes() {
        // NumPy retains this same Python owner, including its resource reservation.
        return py::array(py::dtype::of<std::uint8_t>(),
                         {static_cast<py::ssize_t>(requested_)}, {1}, data_, py::cast(shared_from_this()));
    }
    std::size_t requested_bytes() const { return requested_; }
    std::size_t mapped_bytes() const { return mapped_; }
    int node() const { return node_; }

private:
    void* data_ = MAP_FAILED;
    std::size_t requested_ = 0, mapped_ = 0, page_ = 0;
    int node_ = -1;
};

PYBIND11_MODULE(_numa, module) {
    module.doc() = "Strict mmap/mbind/first-touch/move_pages allocation; no silent fallback";
    module.attr("compiler") = __VERSION__;
    module.attr("placement_policy") = "MPOL_BIND; base pages; complete residency query";
    py::class_<Region, std::shared_ptr<Region>>(module, "Region", py::dynamic_attr())
        .def(py::init<std::size_t, int>(), py::call_guard<py::gil_scoped_release>())
        .def("bytes", &Region::bytes)
        .def("residency", &Region::residency, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("requested_bytes", &Region::requested_bytes)
        .def_property_readonly("mapped_bytes", &Region::mapped_bytes)
        .def_property_readonly("node", &Region::node);
}
