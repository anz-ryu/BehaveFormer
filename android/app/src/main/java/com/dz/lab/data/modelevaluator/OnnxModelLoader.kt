package com.dz.lab.data.modelevaluator

import ai.onnxruntime.OnnxSequence
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import java.io.File
import java.io.FileOutputStream
import java.io.IOException

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

    fun close() {
        session?.close()
        env?.close()
    }
}