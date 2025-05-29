package com.example.lensflare

import android.util.Log
import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.Toast
import androidx.activity.ComponentActivity
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.imgproc.Imgproc
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import java.nio.ByteBuffer
import android.graphics.Bitmap
import org.opencv.core.Core

class MainActivity : ComponentActivity() {
    private var cameraExecutor: ExecutorService? = null
    private var isFlareRemovalEnabled = false
    private var lensFlareAnalyzer: LensFlareAnalyzer? = null
    private lateinit var previewView: PreviewView
    private lateinit var processedImageView: ImageView
    
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted: Boolean ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Permission request denied", Toast.LENGTH_LONG).show()
        }
    }
    
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        Log.d("MainActivity", "Starting onCreate")
        Log.d("OpenCV", "Trying to initialize OpenCV")
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "OpenCV initialization failed")
            Toast.makeText(this, "OpenCV initialization failed", Toast.LENGTH_LONG).show()
            return
        } else {
            Log.d("OpenCV", "OpenCV initialized successfully")
        }

        // Инициализируем views
        previewView = findViewById<PreviewView>(R.id.previewView)
        processedImageView = findViewById<ImageView>(R.id.processedImageView)

        // Настраиваем кнопку переключения режима
        findViewById<Button>(R.id.modeButton).setOnClickListener {
            isFlareRemovalEnabled = !isFlareRemovalEnabled
            it as Button
            it.text = if (isFlareRemovalEnabled) "Режим удаления засветов" else "Обычный режим"
            Log.d("MainActivity", "Режим удаления засветов: ${if (isFlareRemovalEnabled) "включен" else "выключен"}")
            
            // Переключаем видимость элементов
            if (isFlareRemovalEnabled) {
                previewView.visibility = android.view.View.GONE
                processedImageView.visibility = android.view.View.VISIBLE
            } else {
                previewView.visibility = android.view.View.VISIBLE
                processedImageView.visibility = android.view.View.GONE
            }
            
            // Обновляем анализатор с новым режимом
            lensFlareAnalyzer?.updateMode(isFlareRemovalEnabled)
            
            // Показываем уведомление о режиме
            Toast.makeText(
                this,
                if (isFlareRemovalEnabled) "Режим удаления засветов включен" else "Обычный режим",
                Toast.LENGTH_SHORT
            ).show()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            Log.d("MainActivity", "Camera permission already granted")
            startCamera()
        } else {
            Log.d("MainActivity", "Requesting camera permission")
            requestPermissionLauncher.launch(Manifest.permission.CAMERA)
        }
    }

    private fun startCamera() {
        Log.d("Camera", "Starting camera initialization")
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                Log.d("Camera", "Camera provider obtained")
                
                val preview = Preview.Builder().build()
                lensFlareAnalyzer = LensFlareAnalyzer(isFlareRemovalEnabled, applicationContext) { bitmap ->
                    runOnUiThread {
                        processedImageView.setImageBitmap(bitmap)
                        processedImageView.scaleType = ImageView.ScaleType.CENTER_CROP
                    }
                }
                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        val executor = cameraExecutor
                        if (executor != null) {
                            it.setAnalyzer(executor, lensFlareAnalyzer!!)
                        } else {
                            Log.e("Camera", "Camera executor is null")
                            Toast.makeText(this, "Camera initialization failed", Toast.LENGTH_LONG).show()
                        }
                    }
                
                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                Log.d("Camera", "Camera components created")
                
                try {
                    cameraProvider.unbindAll()
                    
                    // Set the preview use case to the PreviewView
                    preview.setSurfaceProvider(previewView.surfaceProvider)
                    
                    cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, imageAnalysis
                    )
                    Log.d("Camera", "Camera successfully bound")
                } catch (exc: Exception) {
                    Log.e("Camera", "Failed to bind camera", exc)
                    Toast.makeText(this, "Failed to bind camera", Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                Log.e("Camera", "Error in camera setup", e)
                Toast.makeText(this, "Error in camera setup: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }
    
    override fun onDestroy() {
        super.onDestroy()
        try {
            lensFlareAnalyzer?.release()
            cameraExecutor?.shutdown()
            cameraExecutor = null
        } catch (e: Exception) {
            Log.e("MainActivity", "Error shutting down camera executor", e)
        }
    }
    
    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}

class LensFlareAnalyzer(
    private var isFlareRemovalEnabled: Boolean,
    private val context: android.content.Context,
    private val onImageProcessed: (Bitmap) -> Unit
) : ImageAnalysis.Analyzer {
    private val processor = LensFlareProcessor(context)
    
    fun updateMode(enabled: Boolean) {
        isFlareRemovalEnabled = enabled
        Log.d("LensFlareAnalyzer", "Mode updated: ${if (enabled) "flare removal enabled" else "normal mode"}")
    }
    
    override fun analyze(imageProxy: ImageProxy) {
        // Обрабатываем изображения только в режиме удаления засветов
        if (!isFlareRemovalEnabled) {
            imageProxy.close()
            return
        }
        
        var bgrMat: Mat? = null
        var processedMat: Mat? = null
        
        try {
            Log.d("LensFlareAnalyzer", "Processing frame with flare removal")

            // Конвертация ImageProxy YUV_420_888 в BGR Mat
            val yPlane = imageProxy.planes[0]
            val uPlane = imageProxy.planes[1]
            val vPlane = imageProxy.planes[2]

            val yBuffer = yPlane.buffer.apply { rewind() }
            val uBuffer = uPlane.buffer.apply { rewind() }
            val vBuffer = vPlane.buffer.apply { rewind() }

            val ySize = yBuffer.remaining()
            
            val imageWidth = imageProxy.width
            val imageHeight = imageProxy.height

            val yuv_I420_data = ByteArray(imageWidth * imageHeight * 3 / 2)

            // 1. Y Plane
            yBuffer.get(yuv_I420_data, 0, ySize)

            // 2. U Plane (Chroma Blue)
            var destOffset = ySize
            val uRowStride = uPlane.rowStride
            val uPixelStride = uPlane.pixelStride
            val chromaWidth = imageWidth / 2
            val chromaHeight = imageHeight / 2
            for (row in 0 until chromaHeight) {
                for (col in 0 until chromaWidth) {
                    yuv_I420_data[destOffset++] = uBuffer.get(row * uRowStride + col * uPixelStride)
                }
            }

            // 3. V Plane (Chroma Red)
            val vRowStride = vPlane.rowStride
            val vPixelStride = vPlane.pixelStride
            for (row in 0 until chromaHeight) {
                for (col in 0 until chromaWidth) {
                    yuv_I420_data[destOffset++] = vBuffer.get(row * vRowStride + col * vPixelStride)
                }
            }

            val yuvMat = Mat(imageHeight * 3 / 2, imageWidth, CvType.CV_8UC1)
            yuvMat.put(0, 0, yuv_I420_data)

            bgrMat = Mat()
            Imgproc.cvtColor(yuvMat, bgrMat, Imgproc.COLOR_YUV2BGR_I420)
            yuvMat.release()
            
            Log.d("LensFlareAnalyzer", "Image converted to BGR: ${bgrMat.cols()}x${bgrMat.rows()}")

            // Поворачиваем изображение на 90 градусов по часовой стрелке
            // Важно: после поворота cols и rows поменяются местами
            Core.rotate(bgrMat, bgrMat, Core.ROTATE_90_CLOCKWISE)
            Log.d("LensFlareAnalyzer", "Image rotated: ${bgrMat.cols()}x${bgrMat.rows()}")
            
            // Обрабатываем изображение
            processedMat = processor.processImage(bgrMat)
            
            // Конвертируем в RGBA для Bitmap
            processedMat?.let { nonNullProcessed ->
                val rgba = Mat()
                Imgproc.cvtColor(nonNullProcessed, rgba, Imgproc.COLOR_BGR2RGBA)
                
                // Конвертируем Mat в Bitmap
                val bitmap = Bitmap.createBitmap(
                    rgba.cols(), // Используем размеры повернутого и обработанного Mat
                    rgba.rows(),
                    Bitmap.Config.ARGB_8888
                )
                Utils.matToBitmap(rgba, bitmap)
                
                // Освобождаем RGBA матрицу
                rgba.release()
                
                // Проверяем, что bitmap не пустой
                Log.d("LensFlareAnalyzer", "Bitmap info: ${bitmap.width}x${bitmap.height}, config: ${bitmap.config}")
                
                // Отправляем Bitmap в UI
                onImageProcessed(bitmap)
                
                Log.d("LensFlareAnalyzer", "Processed bitmap sent to UI: ${bitmap.width}x${bitmap.height}")
            }
        } catch (e: Exception) {
            Log.e("LensFlareAnalyzer", "Error processing frame", e)
            e.printStackTrace()
        } finally {
            bgrMat?.release()
            processedMat?.release()
            imageProxy.close()
        }
    }
    
    fun release() {
        // processor.release() // Удаляем этот вызов, т.к. LensFlareProcessor.release() больше не существует
    }
} 