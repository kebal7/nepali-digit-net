import load

training_data, test_data = load.get_training_and_test_data()

print(f"Training samples: {len(training_data)}")
print(f"Test samples: {len(test_data)}")

