#include "seq/rezantseva_shell_batcher/include/ops_seq_rezantseva.hpp"

#include <vector>

std::vector<double> rezantseva_shell_batcher_seq::ShellBatcherSortSequential::shellSort(const std::vector<double> &v) {
  size_t n = v.size();
  std::vector<double> result(v);

  for (size_t step = n / 2; step > 0; step /= 2) {
    for (int i = step; i < n; i++) {
      for (int j = i - step; j >= 0 && result[j] > result[j + step]; j -= step) {  // Compare elements at a distance
                                                                                   // step
        double tmp = result[j];
        result[j] = result[j + step];
        result[j + step] = tmp;
      }
    }
  }
  return result;
}

bool rezantseva_shell_batcher_seq::ShellBatcherSortSequential::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] > 1 && task_data->inputs_count.size() == 1 &&
         task_data->outputs_count.size() == 1 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool rezantseva_shell_batcher_seq::ShellBatcherSortSequential::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);  // get data
  input_ = std::vector<double>(in_ptr, in_ptr + input_size);        // fill

  unsigned int output_size = task_data->outputs_count[0];
  result_ = std::vector<double>(output_size, 0);
  return true;
}

bool rezantseva_shell_batcher_seq::ShellBatcherSortSequential::RunImpl() {
  result_ = shellSort(input_);
  return true;
}

bool rezantseva_shell_batcher_seq::ShellBatcherSortSequential::PostProcessingImpl() {
  std::copy(result_.begin(), result_.end(), reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
