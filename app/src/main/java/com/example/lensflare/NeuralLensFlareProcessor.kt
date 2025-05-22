package com.example.lensflare

import android.content.Context
import android.graphics.Bitmap
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import ai.onnxruntime.*
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.FloatBuffer

class NeuralLensFlareProcessor(private val context: Context) {
    private var ortSession: OrtSession? = null
    private val inputSize = 256

    init {
        try {
            Log.d("NeuralLensFlareProcessor", "Initializing ONNX Runtime")
            // Инициализируем ONNX Runtime
            val env = OrtEnvironment.getEnvironment()
            val sessionOptions = OrtSession.SessionOptions()
            
            // Загружаем модель
            Log.d("NeuralLensFlareProcessor", "Loading model from assets")
            val modelPath = context.assets.open("lens_flare_model.onnx")
            val modelBytes = modelPath.readBytes()
            Log.d("NeuralLensFlareProcessor", "Model size: ${modelBytes.size} bytes")
            
            ortSession = env.createSession(modelBytes, sessionOptions)
            Log.d("NeuralLensFlareProcessor", "Model loaded successfully")
        } catch (e: Exception) {
            Log.e("NeuralLensFlareProcessor", "Error loading model", e)
            e.printStackTrace()
        }
    }

    fun processImage(input: Bitmap): Bitmap {
        if (ortSession == null) {
            Log.e("NeuralLensFlareProcessor", "ONNX session is null")
            return input
        }

        try {
            Log.d("NeuralLensFlareProcessor", "Starting image processing")
            Log.d("NeuralLensFlareProcessor", "Input image size: ${input.width}x${input.height}")
            
            // Поворачиваем входное изображение на 90 градусов
            val matrix = android.graphics.Matrix()
            matrix.postRotate(90f)
            val rotatedInput = android.graphics.Bitmap.createBitmap(
                input, 0, 0, input.width, input.height, matrix, true
            )
            
            // Конвертируем Bitmap в Mat
            val mat = Mat()
            Utils.bitmapToMat(rotatedInput, mat)
            Log.d("NeuralLensFlareProcessor", "Converted to Mat: ${mat.width()}x${mat.height()}, channels: ${mat.channels()}")
            
            // Изменяем размер
            val resized = Mat()
            Imgproc.resize(mat, resized, org.opencv.core.Size(inputSize.toDouble(), inputSize.toDouble()))
            Log.d("NeuralLensFlareProcessor", "Resized image: ${resized.width()}x${resized.height()}, channels: ${resized.channels()}")
            
            // Нормализуем значения
            val normalized = Mat()
            resized.convertTo(normalized, org.opencv.core.CvType.CV_32F, 1.0/255.0)
            Log.d("NeuralLensFlareProcessor", "Normalized values: ${normalized.width()}x${normalized.height()}, channels: ${normalized.channels()}")
            
            // Подготавливаем входные данные
            val inputData = FloatArray(inputSize * inputSize * 3)
            normalized.get(0, 0, inputData)
            Log.d("NeuralLensFlareProcessor", "Got input data, size: ${inputData.size}")
            
            // Создаем FloatBuffer и копируем данные
            val inputBuffer = FloatBuffer.allocate(inputData.size)
            inputBuffer.put(inputData)
            inputBuffer.rewind()
            
            // Создаем входной тензор
            Log.d("NeuralLensFlareProcessor", "Creating input tensor")
            val inputTensor = OnnxTensor.createTensor(
                OrtEnvironment.getEnvironment(),
                inputBuffer,
                longArrayOf(1, 3, inputSize.toLong(), inputSize.toLong())
            )
            Log.d("NeuralLensFlareProcessor", "Input tensor created: ${inputTensor.info}")
            
            // Запускаем инференс
            Log.d("NeuralLensFlareProcessor", "Running inference")
            val inputs = mapOf("input" to inputTensor)
            val output = ortSession?.run(inputs)
            Log.d("NeuralLensFlareProcessor", "Inference completed")
            
            // Получаем выходной тензор
            val outputTensor = output?.get("output") as? OnnxTensor
            val outputBuffer = outputTensor?.floatBuffer
            Log.d("NeuralLensFlareProcessor", "Got output tensor: ${outputTensor?.info}")
            
            if (outputBuffer == null) {
                Log.e("NeuralLensFlareProcessor", "Output buffer is null")
                return input
            }
            
            // Создаем выходной Mat
            val outputMat = Mat(inputSize, inputSize, org.opencv.core.CvType.CV_32FC3)
            
            // Копируем данные из буфера
            val outputData = FloatArray(inputSize * inputSize * 3)
            outputBuffer.rewind()
            outputBuffer.get(outputData)
            
            // Денормализуем значения
            for (i in outputData.indices) {
                outputData[i] = outputData[i] * 255.0f
            }
            Log.d("NeuralLensFlareProcessor", "Denormalized values")
            
            outputMat.put(0, 0, outputData)
            Log.d("NeuralLensFlareProcessor", "Output Mat created: ${outputMat.width()}x${outputMat.height()}, channels: ${outputMat.channels()}")
            
            // Конвертируем обратно в Bitmap
            val outputBitmap = Bitmap.createBitmap(inputSize, inputSize, Bitmap.Config.ARGB_8888)
            Utils.matToBitmap(outputMat, outputBitmap)
            Log.d("NeuralLensFlareProcessor", "Converted to output Bitmap: ${outputBitmap.width}x${outputBitmap.height}")
            
            // Поворачиваем выходное изображение обратно на -90 градусов
            val outputMatrix = android.graphics.Matrix()
            outputMatrix.postRotate(-90f)
            val finalBitmap = android.graphics.Bitmap.createBitmap(
                outputBitmap, 0, 0, outputBitmap.width, outputBitmap.height, outputMatrix, true
            )
            outputBitmap.recycle()
            
            // Освобождаем ресурсы
            mat.release()
            resized.release()
            normalized.release()
            outputMat.release()
            inputTensor.close()
            outputTensor?.close()
            rotatedInput.recycle()
            Log.d("NeuralLensFlareProcessor", "Resources released")
            
            return finalBitmap
        } catch (e: Exception) {
            Log.e("NeuralLensFlareProcessor", "Error processing image", e)
            e.printStackTrace()
            return input
        }
    }

    fun close() {
        try {
            ortSession?.close()
            Log.d("NeuralLensFlareProcessor", "Session closed")
        } catch (e: Exception) {
            Log.e("NeuralLensFlareProcessor", "Error closing session", e)
            e.printStackTrace()
        }
    }
} 