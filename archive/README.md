# Archive: original 2024 notebooks

These are the exploratory notebooks from the original research period
(March–June 2024), kept unchanged for the record. They were written in Google
Colab, several depend on cells that were later deleted or run out of order, and
their saved outputs do not always match the code. **They are not expected to
run top to bottom.** The maintained versions of this work are
[`classroom/`](../classroom), [`classroom_experiments.ipynb`](../classroom_experiments.ipynb)
and [`backprop_from_scratch.ipynb`](../backprop_from_scratch.ipynb).

| Notebook | What it was | Superseded by |
|---|---|---|
| `NN_multinodal_hidden_layer.ipynb` | Hand-derived backprop for a leaky-ReLU network, 2 then *n* hidden nodes | `backprop_from_scratch.ipynb` |
| `Student_class.ipynb` | First sketch of a `Student` class wrapping that network | `classroom/model.py` |
| `lecture_vs_active_learning.ipynb` | Class of numpy networks trained with `scipy.optimize`; first attempt at group learning | `classroom/strategies.py` |
| `Pytorch_implementation.ipynb` | PyTorch `Student` class and optimizer comparison | `classroom/model.py` |
| `active_learning_ann.ipynb` | Lecture vs. "active learning" via a group's summed loss | `classroom/strategies.py` (`lecture`, `shared_loss_2024`, `group_study`) |
| `LeaderLearning.ipynb` | Activation / optimizer / neuron sweeps, memorization vs. critical-thinking grading, unfinished leader learning | `classroom/strategies.py` (`leader`), `classroom/grading.py` |

## Known issues (found in the 2026 review)

- **The "active learning" in `active_learning_ann.ipynb` did not couple students.** Summing a
  group's losses and back-propagating gives each student exactly its solo gradient, because
  each student's weights appear only in its own term. Its lecture comparison also used a different
  optimizer (SGD vs. Adam). This is reproduced as the `shared_loss_2024` strategy and verified
  in `tests/test_classroom.py`.
- **`lecture_vs_active_learning.ipynb`**: the group's averaged *prediction* was passed to
  `scipy.optimize.minimize` as the initial *weight vector*, which raises
  `ValueError: cannot reshape array of size 1 into shape (1,5)`.
- **`LeaderLearning.ipynb`** calls `polynomial_optimizer_test_training`, which no longer exists
  in the notebook. The leader-learning training loop was left as debugging prints.
- **`Student_class.ipynb` / `NN_multinodal_hidden_layer.ipynb`**: all weights started at 0.5,
  so hidden neurons stayed identical throughout training (see the "symmetry trap" in
  `backprop_from_scratch.ipynb`). Some printouts labelled "Outer Layer Weights" printed the
  hidden weights.

## Attribution

Parts of these notebooks were not written by me. The numpy `NN` class, `concept`, and
`lecture_obj` in `lecture_vs_active_learning.ipynb` came from Prof. Pedro's notebook, and the
PyTorch `Student` class in `Pytorch_implementation.ipynb` was written by my teammate Sankritya.
My own work in that notebook is the section headed "Sierra experimenting with Sankritya's class".
