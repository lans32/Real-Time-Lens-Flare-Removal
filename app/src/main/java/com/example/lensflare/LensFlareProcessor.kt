package com.example.lensflare

import android.util.Log
import org.opencv.core.*
import org.opencv.imgproc.Imgproc
import org.opencv.photo.Photo
import kotlin.math.max

class LensFlareProcessor {
    fun processImage(input: Mat): Mat {
        try {
            Log.d("LensFlareProcessor", "Starting advanced flare processing")
            val result = input.clone()

            // Конвертируем в HSV для анализа яркости
            val hsv = Mat()
            Imgproc.cvtColor(result, hsv, Imgproc.COLOR_BGR2HSV)

            val channels = ArrayList<Mat>()
            Core.split(hsv, channels)
            val valueChannel = channels[2] // V канал (яркость)

            // Определяем порог для ярких областей (потенциальные блики)
            // Это значение, возможно, потребует подстройки
            val brightnessThreshold = 240.0 // Увеличено для меньшей чувствительности
            val flareMask = Mat()
            Imgproc.threshold(valueChannel, flareMask, brightnessThreshold, 255.0, Imgproc.THRESH_BINARY)

            // Немного расширяем маску, чтобы захватить области вокруг бликов
            val kernel = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(7.0, 7.0)) // Уменьшен размер ядра
            Imgproc.dilate(flareMask, flareMask, kernel)

            // Создаем "затемняющий" слой
            // Уменьшаем яркость в областях бликов
            // Коэффициент затемнения, можно настроить (0.0 - очень темно, 1.0 - без изменений)
            // val darkeningFactor = 0.3

            // for (y in 0 until valueChannel.rows()) {
            //     for (x in 0 until valueChannel.cols()) {
            //         if (flareMask.get(y, x)[0] > 0) { // Если это область блика
            //             val currentValue = valueChannel.get(y, x)[0]
            //             val newValue = currentValue * darkeningFactor
            //             valueChannel.put(y, x, newValue)
            //         }
            //     }
            // }

            // Собираем обратно HSV изображение
            // Core.merge(channels, hsv)
            // Imgproc.cvtColor(hsv, result, Imgproc.COLOR_HSV2BGR)

            // Используем inpaint для "закрашивания" бликов
            // Radius of a circular neighborhood of each point inpainted.
            val inpaintRadius = 3.0 // Уменьшен радиус для меньшего размытия
            Photo.inpaint(input, flareMask, result, inpaintRadius, Photo.INPAINT_NS) // Изменен алгоритм на INPAINT_NS

            // Очистка
            hsv.release()
            valueChannel.release()
            flareMask.release()
            kernel.release()
            channels.forEach { it.release() }

            Log.d("LensFlareProcessor", "Advanced flare processing completed")
            return result

        } catch (e: Exception) {
            Log.e("LensFlareProcessor", "Error in advanced flare processing", e)
            return input.clone() // Возвращаем оригинал в случае ошибки
        }
    }
} 