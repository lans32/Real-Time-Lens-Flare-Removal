package com.example.lensflare

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.opencv.core.Mat
import org.pytorch.IValue
import org.pytorch.LiteModuleLoader
import org.pytorch.Module
import org.pytorch.Tensor
import org.pytorch.torchvision.TensorImageUtils
import org.pytorch.MemoryFormat
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.math.min

class LensFlareProcessor(private val context: Context) {
    private var module: Module? = null

    init {
        try {
            // Изменяем имя файла модели
            module = LiteModuleLoader.load(assetFilePath("model_specular_removal.ptl"))
            Log.d("LensFlareProcessor", "Модель specular-removal успешно загружена")
        } catch (e: IOException) {
            Log.e("LensFlareProcessor", "Ошибка загрузки модели specular-removal: ", e)
        }
    }

    fun processImage(bgrMat: Mat): Mat {
        if (module == null) {
            Log.e("LensFlareProcessor", "Модель не загружена, возвращаем исходное изображение")
            return bgrMat
        }

        val originalWidth = bgrMat.cols()
        val originalHeight = bgrMat.rows()

        if (originalWidth == 0 || originalHeight == 0) {
            Log.e("LensFlareProcessor", "Исходное изображение имеет нулевые размеры.")
            return bgrMat
        }

        try {
            // 1. Предобработка: OpenCV Mat -> Bitmap
            val inputBitmapARGB = Bitmap.createBitmap(originalWidth, originalHeight, Bitmap.Config.ARGB_8888)
            org.opencv.android.Utils.matToBitmap(bgrMat, inputBitmapARGB)

            // Определяем целевой размер для модели (224x224 или 448x448)
            // (в infer.py: if img.shape[2] < 250: size=[224,224] else: size=[448,448])
            // Используем высоту для определения размера, как в infer.py (img.shape[2] это высота)
            val targetHeight: Int
            val targetWidth: Int

            // Устанавливаем фиксированный меньший размер для ускорения
            targetHeight = 128
            targetWidth = 128
            /* Закомментируем динамическое определение размера
            if (originalHeight < 250) { // В infer.py используется высота (img.shape[2])
                targetHeight = 224
                targetWidth = 224
            } else {
                targetHeight = 448
                targetWidth = 448
            }
            */
            Log.d("LensFlareProcessor", "Original size: ${originalWidth}x${originalHeight}, Target model size: ${targetWidth}x${targetHeight}")

            // Масштабируем Bitmap до целевого размера
            val scaledBitmap = Bitmap.createScaledBitmap(inputBitmapARGB, targetWidth, targetHeight, true)

            // ЛОГИРОВАНИЕ: Проверим несколько пикселей из scaledBitmap
            /*val BmpW = scaledBitmap.width
            val BmpH = scaledBitmap.height
            if (BmpW > 4 && BmpH > 4) {
                Log.d("LensFlareProcessor_BMP", "scaledBitmap (TopLeft 2x2 pixels, ARGB Int): " +
                        "[${scaledBitmap.getPixel(0,0)}, ${scaledBitmap.getPixel(1,0)}]" +
                        " [${scaledBitmap.getPixel(0,1)}, ${scaledBitmap.getPixel(1,1)}]")
            }*/

            // Конвертируем масштабированный Bitmap в тензор
            // Новый способ: ручная нормализация для получения [0,1]
            val numElementsTensor = 3 * targetHeight * targetWidth
            val floatBuffer = Tensor.allocateFloatBuffer(numElementsTensor)
            
            // Шаг 1: Получить значения пикселей как Float [0-255] используя bitmapToFloatBuffer
            // mean=[0,0,0], std=[1,1,1] должен просто конвертировать байты в float без изменения масштаба.
            TensorImageUtils.bitmapToFloatBuffer(
                scaledBitmap,
                0, 0, targetWidth, targetHeight, 
                floatArrayOf(0.0f, 0.0f, 0.0f), // normMeanRGB
                floatArrayOf(1.0f, 1.0f, 1.0f), // normStdRGB
                floatBuffer, 
                0, 
                MemoryFormat.CHANNELS_LAST // Выход HWC
            )
            
            val hwcFloatArrayOriginalScale = FloatArray(numElementsTensor)
            floatBuffer.rewind()
            floatBuffer.get(hwcFloatArrayOriginalScale)
            // Log.d("LensFlareProcessor_DATA", "HWC FloatArray (0-255 scale, first 15): ${hwcFloatArrayOriginalScale.sliceArray(0..14).contentToString()}")

            // Шаг 2: Ручная нормализация [0-255] -> [0,1] и конвертация HWC -> CHW
            val chwFloatArrayNormalized = FloatArray(numElementsTensor)
            for (h in 0 until targetHeight) {
                for (w in 0 until targetWidth) {
                    for (c in 0 until 3) {
                        val hwcIndex = (h * targetWidth + w) * 3 + c
                        val chwIndex = (c * targetHeight + h) * targetWidth + w
                        // Нормализуем делением на 255.0f
                        chwFloatArrayNormalized[chwIndex] = hwcFloatArrayOriginalScale[hwcIndex]
                    }
                }
            }

            val inputTensor = Tensor.fromBlob(chwFloatArrayNormalized, longArrayOf(1, 3, targetHeight.toLong(), targetWidth.toLong()))

            Log.d("LensFlareProcessor", "Input tensor created with shape: ${inputTensor.shape().contentToString()}")
            // ЛОГИРОВАНИЕ: Проверим новый inputData
            val newTempInputData = inputTensor.dataAsFloatArray
            // Log.d("LensFlareProcessor_DATA", "New Input Tensor Data (first 15 after manual norm): ${newTempInputData.sliceArray(0..14).contentToString()}")

            // 2. Запуск модели
            Log.d("LensFlareProcessor", "Запуск модели specular-removal...")
            // Модель возвращает кортеж из трех тензоров: (coarse_out, second_out, HF)
            val outputTuple = module?.forward(IValue.from(inputTensor))?.toTuple()
            Log.d("LensFlareProcessor", "Модель specular-removal выполнена.")

            if (outputTuple == null || outputTuple.size != 3) {
                Log.e("LensFlareProcessor", "Выход модели не является кортежем из 3 элементов. Output: $outputTuple")
                return bgrMat
            }

            val coarseOutTensor = outputTuple[0].toTensor()
            val secondOutTensor = outputTuple[1].toTensor()
            val hfTensor = outputTuple[2].toTensor()

            Log.d("LensFlareProcessor", "Output tensors extracted: coarse_out shape: ${coarseOutTensor.shape().contentToString()}, second_out shape: ${secondOutTensor.shape().contentToString()}, hf shape: ${hfTensor.shape().contentToString()}")


            // 3. Постобработка: Tensor -> Bitmap -> Mat
            // Формула: final_out = input_tensor * (1 - HF_tensor) + second_out_tensor * HF_tensor
            // Все тензоры (inputTensor, secondOutTensor, hfTensor) должны быть в диапазоне [0,1]
            // и иметь одинаковую форму [1, 3, targetHeight, targetWidth]

            // Конвертируем тензоры в FloatArray для ручных операций
            val inputData = inputTensor.dataAsFloatArray // values are [0,1]
            val secondOutData = secondOutTensor.dataAsFloatArray // values are [0,1] (clamped in model)
            val hfData = hfTensor.dataAsFloatArray // values are [0,1] (HFE output usually is, or needs clamping/sigmoid)

            // ЛОГИРОВАНИЕ: Выведем несколько значений из каждого тензора
            /*
            Log.d("LensFlareProcessor_DATA", "Input Tensor (first 5): ${inputData.sliceArray(0..4).contentToString()}")
            Log.d("LensFlareProcessor_DATA", "SecondOut Tensor (first 5): ${secondOutData.sliceArray(0..4).contentToString()}")
            Log.d("LensFlareProcessor_DATA", "HF Tensor (first 5): ${hfData.sliceArray(0..4).contentToString()}")
            // Статистика по HF Tensor
            val hfMin = hfData.minOrNull() ?: 0.0f
            val hfMax = hfData.maxOrNull() ?: 0.0f
            val hfAvg = hfData.average().toFloat()
            Log.d("LensFlareProcessor_DATA", "HF Tensor Stats: Min=$hfMin, Max=$hfMax, Avg=$hfAvg")
            */

            val numElements = (1 * 3 * targetHeight * targetWidth).toInt()
            val finalPixelData = FloatArray(numElements)

            if (inputData.size != numElements || secondOutData.size != numElements || hfData.size != numElements) {
                Log.e("LensFlareProcessor", "Размеры массивов данных тензоров не совпадают! " +
                        "Input: ${inputData.size}, SecondOut: ${secondOutData.size}, HF: ${hfData.size}, Expected: $numElements")
                return bgrMat
            }
            
            Log.d("LensFlareProcessor", "Performing post-processing on CPU...")
            // ТЕСТ 1: Просто выводим inputData, чтобы проверить путь отображения
            Log.d("LensFlareProcessor_TEST", "ТЕСТ 1: Копируем inputData в finalPixelData")
            // inputData.copyInto(finalPixelData) // Закомментировать эту строку
            // Оригинальная постобработка
                    for (i in 0 until numElements) { // Раскомментировать этот блок
                val hfVal = hfData[i] 
                val clampedHfVal = min(1.0f, kotlin.math.max(0.0f, hfVal))
                finalPixelData[i] = inputData[i] * (1.0f - clampedHfVal) + secondOutData[i] * clampedHfVal
            } // Раскомментировать эту строку
            

            // ЛОГИРОВАНИЕ: Выведем несколько значений из finalPixelData до зажима
            // Log.d("LensFlareProcessor_DATA", "FinalPixelData (before clamp, first 5): ${finalPixelData.sliceArray(0..4).contentToString()}")

            // Убедимся, что значения в finalPixelData находятся в диапазоне [0,1]
            for (i in 0 until numElements) {
                finalPixelData[i] = min(1.0f, kotlin.math.max(0.0f, finalPixelData[i]))
            }

            // Создаем Bitmap из FloatArray (данные в [0,1])
            // TensorImageUtils.formatFloatTensorForBitmap ожидает данные CHW
            val outputBitmap = Bitmap.createBitmap(targetWidth, targetHeight, Bitmap.Config.ARGB_8888)
            
            // Конвертация FloatArray ([0,1]) в IntArray пикселей для Bitmap
            val outputPixels = IntArray(targetWidth * targetHeight)
            val H = targetHeight
            val W = targetWidth

            for (h_idx in 0 until H) {
                for (w_idx in 0 until W) {
                    // Индексы для CHW формата:
                    // Исправленная логика индексации
                    val planeSize = H * W
                    val pixelOffsetInPlane = h_idx * W + w_idx
                    val r_idx = pixelOffsetInPlane // Канал R (индекс 0)
                    val g_idx = planeSize + pixelOffsetInPlane // Канал G (индекс 1)
                    val b_idx = 2 * planeSize + pixelOffsetInPlane // Канал B (индекс 2)

                    val r_float = finalPixelData[r_idx] * 255.0f
                    val g_float = finalPixelData[g_idx] * 255.0f
                    val b_float = finalPixelData[b_idx] * 255.0f
                    
                    // Добавляем явный зажим значений в диапазоне [0, 255] перед преобразованием в Int
                    val r = kotlin.math.max(0.0f, kotlin.math.min(255.0f, r_float)).toInt()
                    val g = kotlin.math.max(0.0f, kotlin.math.min(255.0f, g_float)).toInt()
                    val b = kotlin.math.max(0.0f, kotlin.math.min(255.0f, b_float)).toInt()
                    
                    outputPixels[h_idx * W + w_idx] = (0xFF shl 24) or (r shl 16) or (g shl 8) or b
                }
            }
            // ЛОГИРОВАНИЕ: Выведем несколько первых пикселей (в формате Int)
            // Log.d("LensFlareProcessor_DATA", "OutputPixels (first 5): ${outputPixels.sliceArray(0..4).contentToString()}")

            outputBitmap.setPixels(outputPixels, 0, targetWidth, 0, 0, targetWidth, targetHeight)
            Log.d("LensFlareProcessor", "Output bitmap created from processed float array.")

            // Масштабируем результат обратно до исходного размера
            val finalScaledBitmap = Bitmap.createScaledBitmap(outputBitmap, originalWidth, originalHeight, true)
            Log.d("LensFlareProcessor", "Output bitmap scaled to original size: ${finalScaledBitmap.width}x${finalScaledBitmap.height}")

            // Конвертируем Bitmap обратно в Mat
            val resultMat = Mat()
            org.opencv.android.Utils.bitmapToMat(finalScaledBitmap, resultMat)
            Log.d("LensFlareProcessor", "Final Mat created. Type: ${resultMat.type()}, Channels: ${resultMat.channels()}")

            // Освобождаем память от Bitmap'ов
            inputBitmapARGB.recycle()
            scaledBitmap.recycle()
            outputBitmap.recycle()
            finalScaledBitmap.recycle()

            return resultMat

        } catch (e: Exception) {
            Log.e("LensFlareProcessor", "Ошибка при обработке изображения моделью specular-removal: ", e)
            return bgrMat // Возвращаем исходное изображение в случае ошибки
        }
    }

    @Throws(IOException::class)
    private fun assetFilePath(assetName: String): String {
        val file = File(context.filesDir, assetName)
        if (file.exists() && file.length() > 0) {
            return file.absolutePath
        }
        context.assets.open(assetName).use { inputStream ->
            FileOutputStream(file).use { outputStream ->
                val buffer = ByteArray(4 * 1024)
                var read: Int
                while (inputStream.read(buffer).also { read = it } != -1) {
                    outputStream.write(buffer, 0, read)
                }
                outputStream.flush()
            }
        }
        return file.absolutePath
    }

    fun release() {
        module?.destroy()
        module = null
        Log.d("LensFlareProcessor", "Модуль specular-removal освобожден")
    }
} 