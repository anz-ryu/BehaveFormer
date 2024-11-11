package com.dz.lab.data.modelevaluator

import ai.onnxruntime.OnnxSequence
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.util.Log
import java.io.File
import java.io.FileOutputStream
import java.io.IOException
import kotlin.random.Random
import java.nio.FloatBuffer

class OnnxModelLoader(
    private val context: Context,
    private val modelName: String
) {
    private var env: OrtEnvironment? = null
    private var session: OrtSession? = null

    init {
        env = OrtEnvironment.getEnvironment()
        val modelPath = copyAssetToFile(context, modelName)
        session = env?.createSession(modelPath)
    }

    fun evaluate(input: FloatArray, shape: LongArray): FloatArray {
        session?.let { session ->
            try {
                // 入力テンソルを作成
                val inputTensor = OnnxTensor.createTensor(
                    OrtEnvironment.getEnvironment(),
                    FloatBuffer.wrap(input),
                    shape
                )

                // 推論を実行
                val output = session.run(mapOf("input" to inputTensor))
                
                // 出力結果を取得
                val outputTensor = output[0].value as FloatArray
                return outputTensor
            } catch (e: Exception) {
                throw RuntimeException("モデル評価に失敗しました: ${e.message}")
            }
        } ?: throw RuntimeException("モデルが正しく読み込まれていません")
    }

    @Throws(IOException::class)
    private fun copyAssetToFile(context: Context, fileName: String): String {
        val file = File(context.filesDir, fileName)
        context.assets.open(fileName).use { input ->
            FileOutputStream(file).use { output ->
                val buffer = ByteArray(4096)
                var read: Int
                while (input.read(buffer).also { read = it } != -1) {
                    output.write(buffer, 0, read)
                }
            }
        }
        return file.absolutePath
    }

    fun test(shape: LongArray) {
        try {
            // テスト用の入力データを作成 (1, 50, 10)の形状
            val inputSize = shape.iterator().let { t ->
                var rt = 1
                while (t.hasNext()) {
                    rt *= t.next().toInt()
                }
                rt
            }
            val inputData = FloatArray(inputSize) { Random.nextFloat() }
            
            // FloatBufferを使用して入力テンソルを作成
            val inputTensor = OnnxTensor.createTensor(
                OrtEnvironment.getEnvironment(),
                FloatBuffer.wrap(inputData),
                shape
            )

            // 推論実行
            val output = session?.run(mapOf("input" to inputTensor))
            
            // 結果の処理
            output?.get(0)?.let { tensor ->
                when (val value = tensor.value) {
                    is FloatArray -> {
                        Log.d("OnnxModelLoader", "推論結果 (FloatArray): ${value.contentToString()}")
                    }
                    is Array<*> -> {
                        // 多次元配列の場合の処理
                        val result = StringBuilder("推論結果:\n")
                        when {
                            value.isArrayOf<FloatArray>() -> {
                                (value as Array<FloatArray>).forEachIndexed { i, arr ->
                                    result.append("[$i]: ${arr.contentToString()}\n")
                                }
                            }
                            value.isArrayOf<Array<*>>() -> {
                                (value as Array<Array<*>>).forEachIndexed { i, arr ->
                                    result.append("[$i]: ${arr.contentDeepToString()}\n")
                                }
                            }
                        }
                        Log.d("OnnxModelLoader", result.toString())
                    }
                    else -> Log.d("OnnxModelLoader", "推論結果 (その他): $value")
                }
            }
            
            Log.d("OnnxModelLoader", "テスト実行が成功しました")
        } catch (e: Exception) {
            Log.e("OnnxModelLoader", "テスト実行中にエラーが発生しました: ${e.message}", e)
        }
    }

    fun close() {
        session?.close()
        env?.close()
    }
}