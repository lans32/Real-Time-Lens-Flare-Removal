package com.example.lensflare

import android.util.Log
import org.opencv.core.*
import org.opencv.imgproc.Imgproc

class LensFlareProcessor {
    fun processImage(input: Mat): Mat {
        Log.d("LensFlareProcessor", "Starting image processing")
        Log.d("LensFlareProcessor", "Input image size: ${input.width()}x${input.height()}, channels: ${input.channels()}")
        
        var gray: Mat? = null
        var mask: Mat? = null
        var inverseMask: Mat? = null
        val result = input.clone()
        
        try {
            // Конвертируем в серый для обнаружения ярких областей
            gray = Mat()
            Imgproc.cvtColor(input, gray, Imgproc.COLOR_BGR2GRAY)
            Log.d("LensFlareProcessor", "Converted to grayscale")
            
            // Применяем пороговую обработку с меньшим значением
            mask = Mat()
            Imgproc.threshold(gray, mask, 150.0, 255.0, Imgproc.THRESH_BINARY)
            Log.d("LensFlareProcessor", "Applied threshold")
            
            // Применяем размытие к маске
            Imgproc.GaussianBlur(mask, mask, Size(15.0, 15.0), 0.0)
            Log.d("LensFlareProcessor", "Applied Gaussian blur")
            
            // Создаем инвертированную маску
            inverseMask = Mat()
            Core.bitwise_not(mask, inverseMask)
            Log.d("LensFlareProcessor", "Created inverse mask")
            
            // Применяем маску к изображению
            Core.bitwise_and(input, input, result, inverseMask)
            Log.d("LensFlareProcessor", "Applied mask to image")
            
            // Проверяем размеры результата
            Log.d("LensFlareProcessor", "Result image size: ${result.width()}x${result.height()}, channels: ${result.channels()}")
            
            return result
        } catch (e: Exception) {
            Log.e("LensFlareProcessor", "Error in image processing", e)
            result.release()
            throw e
        } finally {
            gray?.release()
            mask?.release()
            inverseMask?.release()
        }
    }
} 