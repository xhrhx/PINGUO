import numpy as np
import tensorflow as tf
import cv2
import os
from PIL import Image


# global Var
model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)

def read_and_preprocess_images_from_directory(path):
    images = []
    for filename in os.listdir(path):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(path, filename))
            assert img is not None and img.size > 0, f"{filename} not loaded or empty"
            img = cv2.resize(img, (224, 224))  # 注意这里我们将图像缩放至(224, 224)的大小
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
            img = img.astype(np.float32)  # 转换为浮点类型
            img = img / 255.0  # 将像素值缩放至[0, 1]的范围
            images.append(img)
    images = np.stack(images, axis=0)
    return images

def extract_features(images):
    features = model.predict(images)
    return features

def vector_norm(x):
    return x / np.sqrt(np.sum(np.square(x)))

def cosine_similarity(a, b):
    a = vector_norm(np.mean(np.mean(a, axis=1), axis=1))
    b = vector_norm(np.mean(np.mean(b, axis=1), axis=1))
    return np.dot(a.flatten(), b.flatten())


def search_similar_image(in_img: Image, count: int) -> list:

    # load image
    cv_image = cv2.cvtColor(np.array(in_img), cv2.COLOR_RGB2BGR)
    query = read_and_preprocess_images_from_directory(cv_image)

    # load all images in photo-repo
    image_dir = "data"
    image_features = {}
    for filename in os.listdir(image_dir):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img = cv2.imread(os.path.join(image_dir, filename))
            assert img is not None and img.size > 0, f"{filename} not loaded or empty"
            img = cv2.resize(img, (224, 224))  # 注意这里我们将图像缩放至(224, 224)的大小
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 将BGR转为RGB
            img = img.astype(np.float32)  # 转换为浮点类型
            img = img / 255.0  # 将像素值缩放至[0, 1]的范围
            feats = extract_features(np.expand_dims(img, axis=0))
            image_features[filename] = feats

    # query = cv2.imread("C:\\XHRHX\\bishe\\moxing\\input.png")
    # assert query is not None and query.size > 0, "Query image not loaded or empty"
    query = cv2.resize(query, (224, 224))
    query = cv2.cvtColor(query, cv2.COLOR_BGR2RGB)
    query = query.astype(np.float32)
    query = query / 255.0

    query_feats = extract_features(np.expand_dims(query, axis=0))

    similarities = {}
    for filename, feats in image_features.items():
        sim = cosine_similarity(query_feats, feats)
        similarities[filename] = sim

    sorted_similarities = sorted(
        similarities.items(), key=lambda x: x[1], reverse=True)[:count]

    re_images = []
    for filename, sim in sorted_similarities:
        re_images.append(
            Image.open((os.path.join(image_dir, filename))))
        # img = cv2.imread(os.path.join(image_dir, filename))
        # assert img is not None and img.size > 0, f"{filename} not loaded or empty"
        # cv2.imshow(filename, img)

    print('****************')
    print(re_images)
    return re_images
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == 'name':
    print('****************')
    print(search_similar_image(Image.open('.\\moxing\\input.png'), 4))

