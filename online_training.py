import tensorflow as tf
import numpy as np

# Generate some example data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 * X + 1 + 0.1 * np.random.randn(100, 1)

# Define the model as a subclass of tf.keras.Model
class LinearModel(tf.keras.Model):
    def __init__(self):
        super(LinearModel, self).__init__()
        self.dense = tf.keras.layers.Dense(1)

    def call(self, inputs):
        return self.dense(inputs)

# Create an instance of the model
model = LinearModel()

# Define the loss function (mean squared error)
loss_fn = tf.keras.losses.MeanSquaredError()

# Define the optimizer
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01)

# Online learning loop
num_epochs = 10
while(1):
    for epoch in range(num_epochs):
        for i in range(len(X)):
            input_data = X[i:i+1]
            target_data = y[i:i+1]
            
            # Compute gradients and update weights using the current data point
            with tf.GradientTape() as tape:
                predictions = model(input_data)
                current_loss = loss_fn(target_data, predictions)
            gradients = tape.gradient(current_loss, model.trainable_variables)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            
        # Calculate and print the loss for this epoch
        epoch_loss = loss_fn(y, model(X)).numpy()
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {epoch_loss:.4f}")
    
    # After training, you can use the trained model for predictions
    new_data_point = np.array([[0.8]])
    prediction = model(new_data_point).numpy()
    print(f"Prediction for new data point: {prediction[0][0]:.4f}")
