#! /usr/bin/env python
# coding: utf-8
import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN

# Параметры
# ==================================================

# Параметры оценки
tf.flags.DEFINE_integer("batch_size", 64, "Размер пакета (по умолчанию: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Каталог чекпоинтов от тестового прогона")

# Остальные параметры
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Разрешить польщоватся удобным процессором")
tf.flags.DEFINE_boolean("log_device_placement", False, "Залогировать на каком процессоре началось выполнение")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nПараметры:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Загрузка данных. Свои данные можно загрузить сдесь
print("Загрузка данных...")
x_test, y_test, vocabulary, vocabulary_inv = data_helpers.load_data()
y_test = np.argmax(y_test, axis=1)
print("Размер словаря: {:d}".format(len(vocabulary)))
print("Размер тестового набора {:d}".format(len(y_test)))

print("\nОценка...\n")

# Оценка
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Загрузка сохраненного мета графа и восстановление переменных
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Получаем метки из графа по имени
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Тензор, который мы хотим оценить
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Генерируем пакеты для одной попытки
        batches = data_helpers.batch_iter(x_test, FLAGS.batch_size, 1, shuffle=False)

        # Сбор прогнозов
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Вывести точность
correct_predictions = float(sum(all_predictions == y_test))
print("Общее число тестовых примеров: {}".format(len(y_test)))
print("Точность: {:g}".format(correct_predictions/float(len(y_test))))
