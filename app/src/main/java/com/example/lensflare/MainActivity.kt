package com.example.lensflare

import android.util.Log
import android.Manifest
import android.content.pm.PackageManager
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
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : ComponentActivity() {
    private var cameraExecutor: ExecutorService? = null
    
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

        Log.d("OpenCV", "Trying to initialize OpenCV")
        if (!OpenCVLoader.initDebug()) {
            Log.e("OpenCV", "OpenCV initialization failed")
            Toast.makeText(this, "OpenCV initialization failed", Toast.LENGTH_LONG).show()
            return
        } else {
            Log.d("OpenCV", "OpenCV initialized successfully")
        }

        cameraExecutor = Executors.newSingleThreadExecutor()

        if (ContextCompat.checkSelfPermission(
                this,
                Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED
        ) {
            startCamera()
        } else {
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
                val imageAnalysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also {
                        val executor = cameraExecutor
                        if (executor != null) {
                            it.setAnalyzer(executor, LensFlareAnalyzer())
                        } else {
                            Log.e("Camera", "Camera executor is null")
                            Toast.makeText(this, "Camera initialization failed", Toast.LENGTH_LONG).show()
                        }
                    }
                
                val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
                Log.d("Camera", "Camera components created")
                
                try {
                    cameraProvider.unbindAll()
                    
                    // Find the PreviewView
                    val previewView = findViewById<PreviewView>(R.id.previewView)
                    if (previewView == null) {
                        Log.e("Camera", "PreviewView not found")
                        Toast.makeText(this, "Camera preview view not found", Toast.LENGTH_LONG).show()
                        return@addListener
                    }
                    
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

class LensFlareAnalyzer : ImageAnalysis.Analyzer {
    private val processor = LensFlareProcessor()
    
    override fun analyze(imageProxy: ImageProxy) {
        var mat: Mat? = null
        var processed: Mat? = null
        
        try {
            Log.d("ImageAnalysis", "Starting image analysis")
            val buffer = imageProxy.planes[0].buffer
            val data = ByteArray(buffer.remaining())
            buffer.get(data)
            
            mat = Mat(imageProxy.height, imageProxy.width, CvType.CV_8UC4)
            mat.put(0, 0, data)
            Log.d("ImageAnalysis", "Image converted to Mat")
            
            // Конвертируем из RGBA в BGR
            Imgproc.cvtColor(mat, mat, Imgproc.COLOR_RGBA2BGR)
            
            // Обрабатываем изображение
            processed = processor.processImage(mat)
            Log.d("ImageAnalysis", "Image processed")
            
            // Конвертируем обратно в RGBA
            Imgproc.cvtColor(processed, processed, Imgproc.COLOR_BGR2RGBA)
            
            // Проверяем размеры
            val processedData = ByteArray(processed.total().toInt() * processed.channels())
            processed.get(0, 0, processedData)
            
            // Проверяем, что размеры совпадают
            if (processedData.size != data.size) {
                Log.e("ImageAnalysis", "Size mismatch: processed=${processedData.size}, original=${data.size}")
                return
            }
            
            // Проверяем размер буфера перед записью
            if (buffer.remaining() < processedData.size) {
                Log.e("ImageAnalysis", "Buffer overflow: buffer.remaining=${buffer.remaining()}, data.size=${processedData.size}")
                return
            }
            
            // Копируем обработанные данные обратно в буфер
            buffer.rewind()
            buffer.put(processedData)
            
            Log.d("ImageAnalysis", "Image analysis completed successfully")
        } catch (e: Exception) {
            Log.e("ImageAnalysis", "Error in image analysis", e)
            e.printStackTrace()
        } finally {
            mat?.release()
            processed?.release()
            imageProxy.close()
        }
    }
} 