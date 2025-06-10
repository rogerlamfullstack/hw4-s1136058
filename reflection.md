**Which framework felt more intuitive—and why?**

TensorFlow (Keras) felt more intuitive because it offers a high-level API that makes model creation, training, and logging easier. The syntax is simpler and more readable, especially for quick experiments. PyTorch provides more control but needs more code.

**Did mitigation harm accuracy? Is that acceptable here?**

Yes, applying fairness mitigation like Reweighing caused a small drop in accuracy. That’s expected. In high-stakes domains like criminal justice, it’s acceptable because reducing unfair bias is more important than a tiny loss in performance.

**One ethical consideration you’d raise if deploying your CIFAR model in the real world.**

If the CIFAR model is used in real-world applications (like sorting images or surveillance), we must ensure the model doesn't inherit or amplify bias from the training data. It should be tested across diverse groups to avoid unfair treatment based on race, gender, or background.

**Which fairness definition improved more? Any trade-off with accuracy?**

After using the Reweighing method, the **Statistical Parity Difference (SPD)** got better more than the **Equalized Odds Difference (EOD)**. This means the method helped reduce unfair differences in positive outcomes between races.

But, the accuracy dropped a little. That’s a common side effect when trying to make models fairer. In this case, it's okay because fairness is more important in systems like criminal justice. Any trade-off with accuracy?**

After using the Reweighing method, the **Statistical Parity Difference (SPD)** got better more than the **Equalized Odds Difference (EOD)**. This means the method helped reduce unfair differences in positive outcomes between races.

But, the accuracy dropped a little. That’s a common side effect when trying to make models fairer. In this case, it's okay because fairness is more important in systems like criminal justice.After applying the Reweighing mitigation technique, the **Statistical Parity Difference (SPD)** showed a greater improvement compared to the **Equalized Odds Difference (EOD)**. This indicates that the mitigation was more effective at reducing overall bias in the distribution of positive predictions across races.

However, this improvement came with a **slight drop in accuracy**, which is a common trade-off in fairness interventions. In this context, such a trade-off is acceptable because the primary goal is to reduce systemic bias in high-stakes decision-making systems like criminal justice risk assessments.
