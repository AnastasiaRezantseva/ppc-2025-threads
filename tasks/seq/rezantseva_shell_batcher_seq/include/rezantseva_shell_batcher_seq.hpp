#pragma once

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace rezantseva_shell_batcher_seq {

class ShellBatcherSortSequential : public ppc::core::Task {
 public:
  explicit ShellBatcherSortSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, result_;
  std::vector<double> shellSort(const std::vector<double>& v);
};

}  // namespace rezantseva_shell_batcher_seq