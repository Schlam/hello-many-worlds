import cirq, random, sympy
import numpy as np
import tensorflow as tf
import tensowrflow_quantum as tfq


qubit = cirq.GridQubit(0,0)

# Quantum data labels
expected_labels = np.array([[1,0],[0,1]])

# Random rotation of X and Z axes
angle = np.random.uniform(0, 2*np.pi)


# Build the quantum data
a = cirq.Circuit(cirq.Ry(angle)(qubit))
b = cirq.Circuit(cirq.Ry(angle + np.pi/2)(qubit))

quantum_data = tfq.convert_to_tensor([a, b])

# Build the quantum model
q_data_input = tf.keras.Input(shape=(), dtype=tf.dtypes.string)


theta = sympyq_model = cirq.Circuit(cirq.Ry(theta)(qubit))
q_model = cirq.Cirquit(cirq.Ry(theta)(qubit))

expectation = tfq.layers.PQC(q_model, cirq.Z(qubit))
expectation_output = expectation(q_data_input)

# Attach the classical softmax classifier
classifier = tf.keras.layers.Dense(2, activation - tf.keras.activations.softmax)
classifier_output = classifier(expectation_output)



# Train the hybrid model
model = tf.keras.Model(inputs=q_data_input,
outputs=classifier_output)

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=0.1),
    loss=tf.keras.losses.CategoricalCrossentropy())
)

history = model.fit(x=quantum_data, y=expected_labels, epochs - 250, verbose=1)


# Check inference on noisy quantum data points
noise = np.random.uniform(-0.25, 0.25, 2)
test_data = tfq.convert_to_tensor([
    cirq.Circuit(
        cirq.Ry(random_angle + noise[0])(qubit)),
    cirq.Circuit(
        cirq.Ry(random_angle + noise[1])(qubit))
])

predictions = model.predict(test_data)

