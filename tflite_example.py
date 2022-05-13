# save trained model
tf.saved_model.save(model, full_model_save_path)

# load saved model

loaded = tf.saved_model.load(full_model_save_path)

print(list(loaded.signatures.keys()))
infer = loaded.signatures["serving_default"]
print(infer.structured_input_signature)
print(infer.structured_outputs)


## write tf.lite model
tflite_model_save_path = os.path.join(model_save_path, "converted_model.tflite")
with open(tflite_model_save_path, "wb") as f:
    f.write(converter.convert())

## read tf.lite.model 
tflite_model_file = os.path.join(model_save_path, 'converted_model.tflite')
with open(tflite_model_file, 'rb') as fid:
    tflite_model = fid.read()


interpreter = tf.lite.Interpreter(model_content=tflite_model)
interpreter.allocate_tensors()

input_index = interpreter.get_input_details()[0]["index"]
output_index = interpreter.get_output_details()[0]["index"]

for img, label in tqdm(test_batches.take(10)):
    interpreter.set_tensor(input_index, img)
    interpreter.invoke()
    predictions.append(interpreter.get_tensor(output_index))
    test_labels.append(label.numpy()[0])
    test_imgs.append(img)
