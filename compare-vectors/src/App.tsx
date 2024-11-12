import { useState } from 'react'
import './App.css'

function App() {
  // サンプルの64次元ベクトルデータ
  const vectors = [
    Array.from({ length: 64 }, () => Math.random()),
    Array.from({ length: 64 }, () => Math.random()),
    Array.from({ length: 64 }, () => Math.random()),
    Array.from({ length: 64 }, () => Math.random()),
  ]

  const [selectedVectors, setSelectedVectors] = useState<number[]>([])
  const [distance, setDistance] = useState<number | null>(null)

  // ベクトル間のユークリッド距離を計算
  const calculateDistance = (v1: number[], v2: number[]) => {
    return Math.sqrt(
      v1.reduce((sum, val, i) => sum + Math.pow(val - v2[i], 2), 0)
    )
  }

  // ベクトル選択の処理
  const handleVectorSelect = (index: number) => {
    if (selectedVectors.includes(index)) {
      setSelectedVectors(selectedVectors.filter(i => i !== index))
      setDistance(null)
    } else {
      const newSelected = [...selectedVectors, index].slice(-2)
      setSelectedVectors(newSelected)
      
      if (newSelected.length === 2) {
        const dist = calculateDistance(vectors[newSelected[0]], vectors[newSelected[1]])
        setDistance(dist)
      }
    }
  }

  return (
    <div className="container">
      <h1>ベクトル距離計算</h1>
      <div className="vectors-list">
        {vectors.map((vector, index) => (
          <div
            key={index}
            className={`vector-item ${selectedVectors.includes(index) ? 'selected' : ''}`}
            onClick={() => handleVectorSelect(index)}
          >
            ベクトル {index + 1}: [{vector.slice(0, 3).map(v => v.toFixed(2)).join(', ')}...]
          </div>
        ))}
      </div>
      {distance !== null && (
        <div className="distance-result">
          選択されたベクトル間の距離: {distance.toFixed(4)}
        </div>
      )}
    </div>
  )
}

export default App
