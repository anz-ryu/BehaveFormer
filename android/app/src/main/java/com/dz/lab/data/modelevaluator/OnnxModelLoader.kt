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

    fun evaluate(input: FloatArray): FloatArray {
        session?.let { session ->
            try {
                // 入力テンソルを作成
                val inputTensor = OnnxTensor.createTensor(
                    OrtEnvironment.getEnvironment(),
                    arrayOf(input)
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

    fun test() {
        try {
            // テスト用の入力データを作成 (1, 50, 10)の形状
            val inputSize = 1 * 50 * 10
            val inputData = FloatArray(inputSize) { Random.nextFloat() }
            
            // FloatBufferを使用して入力テンソルを作成
            val inputTensor = OnnxTensor.createTensor(
                OrtEnvironment.getEnvironment(),
                FloatBuffer.wrap(inputData),
                longArrayOf(1L, 50L, 10L)
            )

            // 推論実行
            val output = session?.run(mapOf("input" to inputTensor))
            
            // 結果の処理
            output?.get(0)?.let { tensor ->
                when (val value = tensor.value) {
                    is FloatArray -> Log.d("OnnxModelLoader", "推論結果: ${value.contentToString()}")
                    is Array<*> -> Log.d("OnnxModelLoader", "推論結果: ${value.contentToString()}")
                    else -> Log.d("OnnxModelLoader", "推論結果: $value")
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