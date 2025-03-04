#pragma once

#include <omp.h>

#include <utility>
#include <vector>

#include "core/task/include/task.hpp"

namespace rezantseva_shell_batcher_omp {
std::vector<double> ShellSortSeq(const std::vector<double>& v);

class ShellBatcherSortSequential : public ppc::core::Task {
 public:
  explicit ShellBatcherSortSequential(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, result_;
};

class ShellBatcherSortOMP : public ppc::core::Task {
 public:
  explicit ShellBatcherSortOMP(ppc::core::TaskDataPtr task_data) : Task(std::move(task_data)) {}
  bool PreProcessingImpl() override;
  bool ValidationImpl() override;
  bool RunImpl() override;
  bool PostProcessingImpl() override;

 private:
  std::vector<double> input_, result_;
  std::vector<double> MyBatcherMergeOMP(const std::vector<double>& a, const std::vector<double>& b);
  std::vector<double> MyShellSortOMP(const std::vector<double>& v);
};

}  // namespace rezantseva_shell_batcher_omp