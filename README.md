# Project work on object detection on video
The code is written as a pet project for use in future developments

___
## List of topics: 
1. Ideas for the future
2. YOLO4 + cv2
3. ResNet50 (DL + Keras)
4. Moving average

___
## Description: 
The places of application of each of the methods will be described in the notebook (it will be at the very bottom of the page)

### 1.Ideas for the future
To study the work of transformers. 

It is possible to try combined approaches using models from this file and transformers. 

Work with time series

### 2. YOLO4 + cv2
Important stages:
1. converting image format to blob
```python
    height, width, _ = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608),
                                 (0, 0, 0), swapRB=True, crop=False)
```

2. Object search function
```python
 # Поиск объектов на изображение:
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)
                box = [center_x - obj_width // 2, center_y - obj_height // 2,
                       obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    # Выбор: удаление дублирующийх рамок, подсчёт объектов и отрисовка рамок
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        box_index = box_index
        box = boxes[box_index]
        class_index = class_indexes[box_index]


        if classes[class_index] in classes_to_look_for:
            objects_count += 1 # подсчёт количества объектов
            # отрисовка ограничивающей рамки
            image_to_process = draw_object_bounding_box(image_to_process,
                                                        class_index, box)

    final_image = draw_object_count(image_to_process, objects_count)

```

3. Displaying the number of objects on the screen
```python
    white_color = (255, 255, 255)
    black_outline_color = (0, 0, 0)
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              white_color, width, cv2.LINE_AA)
```

4. setting up and running the model 
```python
		# Загрузка весов YOLO из файлов и настройка сети
    net = cv2.dnn.readNetFromDarknet("Resources/yolov4-tiny.cfg",
                                     "Resources/yolov4-tiny.weights")
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]

    # Загрузка из файла классов объектов, которые YOLO может обнаружить
    with open("Resources/coco.names.txt") as file:
        classes = file.read().split("\n")

    # Определение классов, которым будет присвоен приоритет при поиске по изображению
    # Названия указаны в файле coco.names.txt

    video = input("Path to video (or URL): ")
    look_for = input("What we are looking for: ").split(',')

    # Удаление
    list_look_for = []
    for look in look_for:
        list_look_for.append(look.strip())

    classes_to_look_for = list_look_for

    start_video_object_detection(video)
```

### 3. ResNet50 (DL + Keras)
Important stages:
1. Data preprocessing
```python
# конвертируем данные и метки в массивы NumPy
data = np.array(data)
labels = np.array(labels)
# перекодировка в one-hot
lb = LabelBinarizer()
labels = lb.fit_transform(labels)
# секционировать данные на обучающие и тестовые сплиты, используя 75%
# данные для обучения и оставшиеся 25% для тестирования
(trainX, testX, trainY, testY) = train_test_split(data, labels,
	test_size=0.25, stratify=labels, random_state=42)
```

2. Data augmentation
```python
# инициализируем объект аугментации обучающих данных
trainAug = ImageDataGenerator(
	rotation_range=30,
	zoom_range=0.15,
	width_shift_range=0.2,
	height_shift_range=0.2,
	shear_range=0.15,
	horizontal_flip=True,
	fill_mode="nearest")
# инициализируем объект дополнения валидации/тестирования данных (который
# мы добавим среднее вычитание)
valAug = ImageDataGenerator()
# определяем среднее вычитание ImageNet (в порядке RGB) и устанавливаем параметр
# среднее значение вычитания для каждого из дополнений данных
mean = np.array([123.68, 116.779, 103.939], dtype="float32")
trainAug.mean = mean
valAug.mean = mean
```

3. Replacement of the head layer in ResNet50
```python
# нагрузить сеть ResNet-50, убедившись, что у головки остались наборы слоев FC
# выкл
baseModel = ResNet50(weights="imagenet", include_top=False,
	input_tensor=Input(shape=(224, 224, 3)))
# сконструируем голову модели, которая будет размещена поверх
# базовая модель
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(512, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(len(lb.classes_), activation="softmax")(headModel)
# поместите модель головы FC поверх базовой модели (это станет
# реальную модель мы будем обучать)
model = Model(inputs=baseModel.input, outputs=headModel)
# Пройдитесь по всем слоям в базовой модели и заморозьте их, чтобы они
# *не* обновляться в процессе обучения
for layer in baseModel.layers:
	layer.trainable = False
```

4. Evolution of the network and metrics 
```python
# эволюция сети
print("[INFO] evaluating network...")
predictions = model.predict(x=testX.astype("float32"), batch_size=32)
print(classification_report(testY.argmax(axis=1),
	predictions.argmax(axis=1), target_names=lb.classes_))
# визуализация метрик:
N = args["epochs"]
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy on Dataset")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig(args["plot"])
```

### 4. Moving average
Important stages:

1. Preparing 
```python
# Загрузка обученной модели и бинаризатора меток с диска
print("[INFO] loading model and label binarizer...")
model = load_model(args["model"])
lb = pickle.loads(open(args["label_bin"], "rb").read())
# инициализируем среднее значение изображения для вычитания среднего вместе с параметром
# очередь прогнозов
mean = np.array([123.68, 116.779, 103.939][::1], dtype="float32")
Q = deque(maxlen=args["size"])
```

2. Making a forecast and averaging
```python
	# делаем прогнозы на фрейме и затем обновляем прогнозы очередь
	preds = model.predict(np.expand_dims(frame, axis=0))[0]
	Q.append(preds)
	# выполнить усреднение прогнозов по текущей истории
	# Предыдущие прогнозы
	results = np.array(Q).mean(axis=0)
	i = np.argmax(results)
	label = lb.classes_[i]
```

And here is the original file: 
https://colab.research.google.com/drive/1kqoOOLbQ6ODTKGPuYvGDV4N0A08iS0BV?usp=sharing
