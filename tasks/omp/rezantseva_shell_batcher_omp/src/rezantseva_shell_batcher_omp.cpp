#include "omp/rezantseva_shell_batcher_omp/include/rezantseva_shell_batcher_omp.hpp"

#include <iostream>
#include <vector>

std::vector<double> rezantseva_shell_batcher_omp::ShellSortSeq(const std::vector<double> &v) {
  size_t n = v.size();
  std::vector<double> result(v);

  for (size_t step = n / 2; step > 0; step /= 2) {
    for (size_t i = step; i < n; i++) {
      double temp = result[i];
      size_t j;
      for (j = i; j >= step && result[j - step] > temp; j -= step) {
        result[j] = result[j - step];
      }
      result[j] = temp;
    }
  }

  return result;
}

bool rezantseva_shell_batcher_omp::ShellBatcherSortSequential::ValidationImpl() {
  // Check equality of counts elements
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] > 1 && task_data->inputs_count.size() == 1 &&
         task_data->outputs_count.size() == 1 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool rezantseva_shell_batcher_omp::ShellBatcherSortSequential::PreProcessingImpl() {
  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);  // get data
  input_ = std::vector<double>(in_ptr, in_ptr + input_size);        // fill

  unsigned int output_size = task_data->outputs_count[0];
  result_ = std::vector<double>(output_size, 0);
  return true;
}

bool rezantseva_shell_batcher_omp::ShellBatcherSortSequential::RunImpl() {
  result_ = ShellSortSeq(input_);
  return true;
}

bool rezantseva_shell_batcher_omp::ShellBatcherSortSequential::PostProcessingImpl() {
  std::copy(result_.begin(), result_.end(), reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}
//------------------------------------------------------------------------------------------------------------------//

bool rezantseva_shell_batcher_omp::ShellBatcherSortOMP::ValidationImpl() {
  return task_data->inputs_count[0] > 1 && task_data->outputs_count[0] > 1 && task_data->inputs_count.size() == 1 &&
         task_data->outputs_count.size() == 1 && task_data->inputs_count[0] == task_data->outputs_count[0];
}

bool rezantseva_shell_batcher_omp::ShellBatcherSortOMP::PreProcessingImpl() {  // Init value for input and output
  unsigned int input_size = task_data->inputs_count[0];
  auto *in_ptr = reinterpret_cast<double *>(task_data->inputs[0]);  // get data
  input_ = std::vector<double>(in_ptr, in_ptr + input_size);        // fill

  unsigned int output_size = task_data->outputs_count[0];
  result_ = std::vector<double>(output_size, 0);
  return true;
}

bool rezantseva_shell_batcher_omp::ShellBatcherSortOMP::RunImpl() {
  result_ = MyShellSortOMP(input_);
  return true;
}

bool rezantseva_shell_batcher_omp::ShellBatcherSortOMP::PostProcessingImpl() {
  std::copy(result_.begin(), result_.end(), reinterpret_cast<double *>(task_data->outputs[0]));
  return true;
}

//------------------------------------------------------------------------------------------------------------------//

// Четно-нечетное слияние Батчера
std::vector<double> rezantseva_shell_batcher_omp::ShellBatcherSortOMP::MyBatcherMergeOMP(const std::vector<double> &a,
                                                                                         const std::vector<double> &b) {
  std::vector<double> result(a.size() + b.size());
  std::size_t i = 0, j = 0, k = 0;

  // Слияние двух отсортированных массивов
  while (i < a.size() && j < b.size()) {
    if (a[i] <= b[j]) {
      result[k++] = a[i++];
    } else {
      result[k++] = b[j++];
    }
  }

  // Добавление оставшихся элементов
  while (i < a.size()) {
    result[k++] = a[i++];
  }
  while (j < b.size()) {
    result[k++] = b[j++];
  }

  // Применение четно-нечетного слияния Батчера
  for (int p = 1; p < result.size(); p *= 2) {
#pragma omp parallel for
    for (int idx = 0; idx < result.size(); idx += 1) {
      if (idx % (2 * p) == 0 && idx + p < result.size()) {
        if (result[idx] > result[idx + p]) {
          std::swap(result[idx], result[idx + p]);
        }
      }
    }
  }

  return result;
}

// Параллельная сортировка Шелла с четно-нечетным слиянием Батчера
std::vector<double> rezantseva_shell_batcher_omp::ShellBatcherSortOMP::MyShellSortOMP(const std::vector<double> &v) {
  size_t threads = 4;
  omp_set_num_threads(threads);

  std::vector<double> result(v);
  int v_size = result.size();
  std::size_t delta = v_size / threads;
  std::size_t remainder = v_size % threads;
  std::vector<std::vector<double>> sorted_parts(threads);

#pragma omp parallel shared(result, remainder, sorted_parts)
  {
    std::size_t curr_thread = omp_get_thread_num();

    // Диапазон данных для текущего потока
    std::size_t start = curr_thread * delta + (curr_thread < remainder ? curr_thread : remainder);
    std::size_t end = start + delta + (curr_thread < remainder ? 1 : 0);

    // Локальный вектор для сортировки
    std::vector<double> local_vec(result.begin() + start, result.begin() + end);

    local_vec = ShellSortSeq(local_vec);

    //  Сохранение отсортированной части
    sorted_parts[curr_thread] = local_vec;
  }

  //  Объединение отсортированных частей
  result = sorted_parts[0];
  for (size_t i = 1; i < threads; i++) {
    result = MyBatcherMergeOMP(result, sorted_parts[i]);
  }

  return result;
}