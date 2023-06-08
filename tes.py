for i in num_iterations:
    for img in images:
        resized_image = img.resize((IMG_SIZE, IMG_SIZE))
        ran_x=random.randint(radius+1,IMG_SIZE-radius-1)
        ran_y=random.randint(radius+1,IMG_SIZE-radius-1)
        img_patch = np.copy(resized_image)
        img_patch=img_patch/255.0
        img_patch = tf.convert_to_tensor(img_patch, dtype=tf.float32)
        with tf.GradientTape() as tape:
            tape.watch(patch_value)
            img_patch = tf.convert_to_tensor(img_patch, dtype=tf.float32)
            img_patch=apply_patch(img_patch,patch_value,ran_x,ran_y,radius)
            predictions = model(img_patch[np.newaxis, ...])
            loss = loss_func(predictions)
        gradients = tape.gradient(loss, patch_value)
        gradients = tf.sign(gradients) * learning_rate
        patch_value = patch_value + tf.squeeze(gradients)
        patch_value = tf.clip_by_value(patch_value, 0, 1)