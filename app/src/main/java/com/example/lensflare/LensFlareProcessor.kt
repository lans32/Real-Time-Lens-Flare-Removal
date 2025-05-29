package com.example.lensflare

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import org.opencv.ximgproc.Ximgproc
import kotlin.math.*

class LensFlareProcessor(private val context: Context) {

    fun processImage(bgrMat: Mat): Mat {
        val width = bgrMat.cols()
        val height = bgrMat.rows()
        if (width == 0 || height == 0) {
            Log.e("LensFlareProcessor", "Input Mat is empty")
            return bgrMat
        }

        val originalBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(bgrMat, originalBitmap)

        val tempMat = Mat(height, width, CvType.CV_8UC4) // BGRA matrix from original bitmap
        Utils.bitmapToMat(originalBitmap, tempMat)

        if (tempMat.empty()) {
            Log.e("LensFlareProcessor", "tempMat is empty after bitmap conversion.")
            originalBitmap.recycle()
            return bgrMat
        }

        val sMaxMap = Mat(height, width, CvType.CV_32FC1)
        val lambdaMaxMap = Mat(height, width, CvType.CV_32FC1)
        val highlightMask = Mat(height, width, CvType.CV_8UC1)

        // Step 1: Calculate initial sMaxMap, lambdaMaxMap, and highlightMask
        for (y in 0 until height) {
            for (x in 0 until width) {
                val pixel = originalBitmap.getPixel(x, y)
                val r = Color.red(pixel).toFloat()
                val g = Color.green(pixel).toFloat()
                val b = Color.blue(pixel).toFloat()

                val sumChroma = r + g + b + 1e-6f
                val normR = r / sumChroma
                val normG = g / sumChroma
                val normB = b / sumChroma

                val sMaxVal = maxOf(normR, normG, normB)
                sMaxMap.put(y, x, sMaxVal.toDouble())

                val sMinVal = minOf(normR, normG, normB)
                
                val denLambda = 1.0f - 3.0f * sMinVal
                var lambdaVal = 0.0f
                if (abs(denLambda) > 1e-6f) {
                    val lambdaR = (normR - sMinVal) / denLambda
                    val lambdaG = (normG - sMinVal) / denLambda
                    val lambdaB = (normB - sMinVal) / denLambda
                    lambdaVal = maxOf(lambdaR, lambdaG, lambdaB).coerceIn(0f, 1f)
                }
                lambdaMaxMap.put(y, x, lambdaVal.toDouble())

                val isHighlight = if (sMaxVal > 0.90f) 255 else 0
                highlightMask.put(y, x, isHighlight.toDouble())
            }
        }

        // Step 2: Iterative refinement of sMaxMap using Joint Bilateral Filter
        val sMaxIterMap = sMaxMap.clone()
        val sMaxFiltered = Mat()
        val numIterationsJBF = 3 // Number of iterations for JBF
        val dJBF = 15             // Diameter of pixel neighborhood
        val sigmaColorJBF = 0.1   // Filter sigma in the color space (for lambdaMaxMap)
        val sigmaSpaceJBF = 15.0  // Filter sigma in the coordinate space

        for (i in 0 until numIterationsJBF) {
            Ximgproc.jointBilateralFilter(sMaxIterMap, lambdaMaxMap, sMaxFiltered, dJBF, sigmaColorJBF, sigmaSpaceJBF)
            for (y in 0 until height) {
                for (x in 0 until width) {
                    val currentSmax = sMaxIterMap.get(y, x)[0].toFloat()
                    val filteredSmax = sMaxFiltered.get(y, x)[0].toFloat()
                    sMaxIterMap.put(y, x, max(currentSmax, filteredSmax).toDouble())
                }
            }
        }
        val finalSMaxMap = sMaxIterMap // Renaming for clarity

        // Step 3: Calculate diffuse color and construct result image
        val resultMat = tempMat.clone() // Initialize result with original image (BGRA)

        for (y in 0 until height) {
            for (x in 0 until width) {
                if (highlightMask.get(y, x)[0] == 255.0) { // Process only highlighted pixels
                    val originalPixelData = tempMat.get(y, x) // BGRA order, 0-255 range
                    val bOrig = originalPixelData[0].toFloat()
                    val gOrig = originalPixelData[1].toFloat()
                    val rOrig = originalPixelData[2].toFloat()
                    val aOrig = originalPixelData[3] // Alpha channel

                    val sValue = finalSMaxMap.get(y, x)[0].toFloat().coerceIn(0f,1f)

                    val denominator = 1.0f - 3.0f * sValue
                    if (denominator > 1e-3f) { // Check if S_value < 1/3 (for robust diffuse color estimation)
                        val maxI = maxOf(rOrig, gOrig, bOrig)
                        val sumI = rOrig + gOrig + bOrig
                        
                        if (sumI > 1e-3f) { // Avoid processing black pixels / division by zero
                             val specularMagnitude = (maxI - sValue * sumI) / denominator

                             val rD = (rOrig - specularMagnitude).coerceIn(0f, 255f)
                             val gD = (gOrig - specularMagnitude).coerceIn(0f, 255f)
                             val bD = (bOrig - specularMagnitude).coerceIn(0f, 255f)
                            
                             resultMat.put(y, x, bD.toDouble(), gD.toDouble(), rD.toDouble(), aOrig)
                        }
                        // else, if sumI is near zero, pixel is black, keep original (already in resultMat)
                    }
                    // else, S_value >= 1/3, diffuse color unreliable, keep original pixel (already in resultMat)
                }
            }
        }

        // Convert result (BGRA) to Bitmap and then to finalMat (BGRA for output)
        val finalBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(resultMat, finalBitmap)
        
        val finalMat = Mat()
        Utils.bitmapToMat(finalBitmap, finalMat) // finalMat will be BGRA

        // Release resources
        originalBitmap.recycle()
        finalBitmap.recycle()
        tempMat.release()
        sMaxMap.release()
        lambdaMaxMap.release()
        highlightMask.release()
        finalSMaxMap.release() // This was sMaxIterMap
        sMaxFiltered.release()
        resultMat.release()

        return finalMat
    }

    // clamp function is not used, can be removed if not needed elsewhere
    // private fun clamp(value: Float): Float {
    // return max(0f, min(1f, value))
    // }
} 