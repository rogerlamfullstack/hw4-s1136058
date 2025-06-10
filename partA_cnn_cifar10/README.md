the acc, loss from cifar_tf:
![alt text](<tf_cifar_acc.png>)
the result from cifar_tf: 50 minutes
![alt text](<tf_cifar.png>)
tensorboard:
![alt text](<Screenshot from 2025-06-11 00-25-46.png>)



Test Accuracy: 0.7359
Time running: 40minutes
![alt text](<Screenshot from 2025-06-11 00-55-48.png>)

4. Result and Discussion:
- Performance
| Framework      | Accuracy                                              |
| -------------- | ----------------------------------------------------- |
| **PyTorch**    | `0.7359`                                              |
| **TensorFlow** | `0.7556` (final epoch)                                |

- Speed & Runtime
TensorFlow is lower than Torch, 50 mins with 40 mins, repsectively.
| Feature       | TensorFlow (`cifar_tf.py`) | PyTorch (`cifar_torch.py`) |
| ------------- | -------------------------- | -------------------------- |
| Lines of code | \~30                       | \~73                       |
| Model         | `Sequential()` API         | Explicit `nn.Module` class |
| Training loop | One-liner `model.fit(...)` | Manual training loop       |
TensorFlow is more concise for prototyping via Keras high-level APIs.

PyTorch is more explicit and flexible, better for research/custom training logic.
| Framework      | Notes                                                                                                      |
| -------------- | ---------------------------------------------------------------------------------------------------------- |
| **TensorFlow** | `TensorBoard` sometimes fails unless `%load_ext tensorboard` is run.                                       |
| **PyTorch**    | Requires boilerplate for loops, no built-in logging if not use `ignite`, `lightning`, or `tensorboardX`. |
| **Shared**     | Both frameworks need careful shape checking for CIFAR and label handling.                                  |
