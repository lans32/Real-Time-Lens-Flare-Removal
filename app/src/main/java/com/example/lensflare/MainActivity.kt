package com.example.lensflare

import android.util.Log
import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.os.Bundle
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
import org.opencv.core.Mat
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import android.widget.ImageView
import java.util.concurrent.atomic.AtomicLong
import android.view.View
import android.widget.FrameLayout
import android.os.Handler
import android.os.Looper
import android.graphics.Color
import java.util.concurrent.atomic.AtomicBoolean

class MainActivity : ComponentActivity() {
    private var cameraExecutor: ExecutorService? = null
    private lateinit var neuralProcessor: NeuralLensFlareProcessor
    private lateinit var processedImageView: ImageView
    private var lastProcessedBitmap: Bitmap? = null
    private val mainHandler = Handler(Looper.getMainLooper())
    private val isProcessing = AtomicBoolean(false)
    
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

        Log.d("MainActivity", "Starting initialization")
        Log.d("OpenCV", "Trying to initialize OpenCV")
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "OpenCV initialization failed")
            Toast.makeText(this, "OpenCV initialization failed", Toast.LENGTH_LONG).show()
            return
        } else {
            Log.d("OpenCV", "OpenCV initialized successfully")
        }

        try {
            // Инициализируем нейронный процессор
            Log.d("MainActivity", "Initializing neural processor")
            neuralProcessor = NeuralLensFlareProcessor(this)
            
            // Находим ImageView для отображения обработанного изображения
            processedImageView = findViewById(R.id.processedImageView)
            Log.d("MainActivity", "Found processedImageView")
            
            // Настраиваем ImageView
            processedImageView.scaleType = ImageView.ScaleType.FIT_CENTER
            processedImageView.adjustViewBounds = true
            processedImageView.visibility = View.VISIBLE
            
            // Проверяем видимость после установки
            Log.d("MainActivity", "ImageView visibility after setup: ${processedImageView.visibility}")
            
            // Создаем тестовое изображение для проверки отображения
            val testBitmap = Bitmap.createBitmap(640, 480, Bitmap.Config.ARGB_8888)
            testBitmap.eraseColor(Color.RED)
            
            // Поворачиваем изображение на 90 градусов
            val matrix = android.graphics.Matrix()
            matrix.postRotate(90f)
            val rotatedBitmap = Bitmap.createBitmap(testBitmap, 0, 0, testBitmap.width, testBitmap.height, matrix, true)
            
            processedImageView.setImageBitmap(rotatedBitmap)
            Log.d("MainActivity", "Set rotated test image to ImageView")
            
            // Проверяем, что изображение установлено
            if (processedImageView.drawable == null) {
                Log.e("MainActivity", "Test image not set to ImageView")
            } else {
                Log.d("MainActivity", "Test image set successfully")
            }
            
            cameraExecutor = Executors.newSingleThreadExecutor()
            Log.d("MainActivity", "Created camera executor")

            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
                ) == PackageManager.PERMISSION_GRANTED
            ) {
                Log.d("MainActivity", "Camera permission granted, starting camera")
                startCamera()
            } else {
                Log.d("MainActivity", "Requesting camera permission")
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        } catch (e: Exception) {
            Log.e("MainActivity", "Error in onCreate", e)
            e.printStackTrace()
            Toast.makeText(this, "Error initializing app: ${e.message}", Toast.LENGTH_LONG).show()
        }
    }

    private fun startCamera() {
        Log.d("Camera", "Starting camera initialization")
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            try {
                val cameraProvider = cameraProviderFuture.get()
                Log.d("Camera", "Camera provider obtained")
                
                val preview = Preview.Builder()
                    .setTargetRotation(windowManager.defaultDisplay.rotation)
                    .build()
                val imageAnalysis = ImageAnalysis.Builder()
                    .setTargetRotation(windowManager.defaultDisplay.rotation)
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        val executor = cameraExecutor
                        if (executor != null) {
                            Log.d("Camera", "Setting up image analyzer")
                            it.setAnalyzer(executor, NeuralLensFlareAnalyzer(neuralProcessor) { processedBitmap ->
                                if (isProcessing.get()) {
                                    Log.d("Camera", "Skipping frame - still processing previous one")
                                    return@NeuralLensFlareAnalyzer
                                }
                                
                                isProcessing.set(true)
                                try {
                                    // Создаем копию битмапа для безопасного использования
                                    val bitmapCopy = processedBitmap.copy(processedBitmap.config, true)
                                    
                                    // Сохраняем последнее обработанное изображение
                                    lastProcessedBitmap?.recycle()
                                    lastProcessedBitmap = bitmapCopy
                                    
                                    // Обновляем UI в главном потоке
                                    mainHandler.post {
                                        try {
                                            if (bitmapCopy.isRecycled) {
                                                Log.e("Camera", "Bitmap is recycled")
                                                return@post
                                            }
                                            
                                            // Проверяем видимость ImageView
                                            if (processedImageView.visibility != View.VISIBLE) {
                                                Log.e("Camera", "ImageView is not visible, setting to VISIBLE")
                                                processedImageView.visibility = View.VISIBLE
                                            }
                                            
                                            Log.d("Camera", "Setting bitmap to ImageView: ${bitmapCopy.width}x${bitmapCopy.height}")
                                            processedImageView.setImageBitmap(bitmapCopy)
                                            Log.d("Camera", "ImageView dimensions: ${processedImageView.width}x${processedImageView.height}")
                                            Log.d("Camera", "ImageView visibility: ${processedImageView.visibility}")
                                            Log.d("Camera", "ImageView scaleType: ${processedImageView.scaleType}")
                                            
                                            // Проверяем, что изображение действительно установлено
                                            if (processedImageView.drawable == null) {
                                                Log.e("Camera", "ImageView drawable is null after setting bitmap")
                                            } else {
                                                Log.d("Camera", "ImageView drawable set successfully")
                                            }
                                        } catch (e: Exception) {
                                            Log.e("Camera", "Error setting bitmap to ImageView", e)
                                            e.printStackTrace()
                                        } finally {
                                            isProcessing.set(false)
                                        }
                                    }
                                } catch (e: Exception) {
                                    Log.e("Camera", "Error updating UI", e)
                                    e.printStackTrace()
                                    isProcessing.set(false)
                                }
                            })
                        } else {
                            Log.e("Camera", "Camera executor is null")
                            Toast.makeText(this, "Camera initialization failed", Toast.LENGTH_LONG).show()
                        }
                    }
                
                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                Log.d("Camera", "Camera components created")
                
                try {
                    cameraProvider.unbindAll()
                    
                    val previewView = findViewById<PreviewView>(R.id.previewView)
                    if (previewView == null) {
                        Log.e("Camera", "PreviewView not found")
                        Toast.makeText(this, "Camera preview view not found", Toast.LENGTH_LONG).show()
                        return@addListener
                    }
                    
                    preview.setSurfaceProvider(previewView.surfaceProvider)
                    
                    cameraProvider.bindToLifecycle(
                        this, cameraSelector, preview, imageAnalysis
                    )
                    Log.d("Camera", "Camera successfully bound")
                } catch (exc: Exception) {
                    Log.e("Camera", "Failed to bind camera", exc)
                    exc.printStackTrace()
                    Toast.makeText(this, "Failed to bind camera", Toast.LENGTH_LONG).show()
                }
            } catch (e: Exception) {
                Log.e("Camera", "Error in camera setup", e)
                e.printStackTrace()
                Toast.makeText(this, "Error in camera setup: ${e.message}", Toast.LENGTH_LONG).show()
            }
        }, ContextCompat.getMainExecutor(this))
    }
    
    override fun onDestroy() {
        super.onDestroy()
        try {
            Log.d("MainActivity", "Shutting down")
            cameraExecutor?.shutdown()
            cameraExecutor = null
            neuralProcessor.close()
            lastProcessedBitmap?.recycle()
            Log.d("MainActivity", "Shutdown complete")
        } catch (e: Exception) {
            Log.e("MainActivity", "Error shutting down", e)
            e.printStackTrace()
        }
    }
    
    companion object {
        private const val REQUEST_CODE_PERMISSIONS = 10
        private val REQUIRED_PERMISSIONS = arrayOf(Manifest.permission.CAMERA)
    }
}

class NeuralLensFlareAnalyzer(
    private val processor: NeuralLensFlareProcessor,
    private val onProcessed: (Bitmap) -> Unit
) : ImageAnalysis.Analyzer {
    private var frameCount = AtomicLong(0)
    private val processEveryNFrames = 1 // Обрабатываем каждый 2-й кадр
    
    override fun analyze(imageProxy: ImageProxy) {
        try {
            val currentFrame = frameCount.incrementAndGet()
            
            // Пропускаем кадры, если это не каждый N-й кадр
            if (currentFrame % processEveryNFrames != 0L) {
                imageProxy.close()
                return
            }
            
            Log.d("ImageAnalysis", "Processing frame $currentFrame")
            val bitmap = imageProxy.toBitmap()
            Log.d("ImageAnalysis", "Converted to bitmap: ${bitmap.width}x${bitmap.height}")
            
            val processedBitmap = processor.processImage(bitmap)
            Log.d("ImageAnalysis", "Image processed: ${processedBitmap.width}x${processedBitmap.height}")
            
            // Вызываем callback с обработанным изображением
            onProcessed(processedBitmap)
            
            bitmap.recycle()
            Log.d("ImageAnalysis", "Image analysis completed")
        } catch (e: Exception) {
            Log.e("ImageAnalysis", "Error in image analysis", e)
            e.printStackTrace()
        } finally {
            imageProxy.close()
        }
    }
    
    private fun ImageProxy.toBitmap(): Bitmap {
        try {
            Log.d("ImageAnalysis", "Converting ImageProxy to Bitmap")
            val yBuffer = planes[0].buffer
            val uBuffer = planes[1].buffer
            val vBuffer = planes[2].buffer
            
            val ySize = yBuffer.remaining()
            val uSize = uBuffer.remaining()
            val vSize = vBuffer.remaining()
            
            Log.d("ImageAnalysis", "Buffer sizes: Y=$ySize, U=$uSize, V=$vSize")
            
            val nv21 = ByteArray(ySize + uSize + vSize)
            
            yBuffer.get(nv21, 0, ySize)
            vBuffer.get(nv21, ySize, vSize)
            uBuffer.get(nv21, ySize + vSize, uSize)
            
            val yuvImage = android.graphics.YuvImage(nv21, android.graphics.ImageFormat.NV21, width, height, null)
            val out = java.io.ByteArrayOutputStream()
            yuvImage.compressToJpeg(android.graphics.Rect(0, 0, width, height), 100, out)
            val imageBytes = out.toByteArray()
            Log.d("ImageAnalysis", "Created JPEG bytes: ${imageBytes.size} bytes")
            
            val bitmap = android.graphics.BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.size)
            
            // Поворачиваем изображение на 90 градусов
            val matrix = android.graphics.Matrix()
            matrix.postRotate(90f)
            val rotatedBitmap = android.graphics.Bitmap.createBitmap(
                bitmap, 0, 0, bitmap.width, bitmap.height, matrix, true
            )
            bitmap.recycle()
            
            return rotatedBitmap
        } catch (e: Exception) {
            Log.e("ImageAnalysis", "Error converting ImageProxy to Bitmap", e)
            e.printStackTrace()
            throw e
        }
    }
} 