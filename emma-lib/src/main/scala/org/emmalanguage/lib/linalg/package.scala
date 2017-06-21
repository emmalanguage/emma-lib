/*
 * Copyright © 2017 TU Berlin (emma@dima.tu-berlin.de)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.emmalanguage
package lib

import org.apache.spark.ml.{linalg => spark}

package object linalg {
  type Vector = spark.Vector
  type DVector = spark.DenseVector
  type SVector = spark.SparseVector

  def dense(values: Array[Double]): DVector =
    spark.Vectors.dense(values).asInstanceOf[DVector]

  def zeros(size: Int): DVector =
    spark.Vectors.zeros(size).asInstanceOf[DVector]

  def sparse(size: Int, indices: Array[Int], values: Array[Double]): SVector =
    spark.Vectors.sparse(size, indices, values).asInstanceOf[SVector]

  def sparse(size: Int, elements: Seq[(Int, Double)]): SVector =
    spark.Vectors.sparse(size, elements).asInstanceOf[SVector]

  def sqdist(x: Vector, y: Vector): Double =
    spark.Vectors.sqdist(x, y)

  implicit class DVectorOps(val x: DVector) extends AnyVal {
    def +=(y: DVector): DVector = {
      BLAS.axpy(1.0, y, x)
      x
    }

    def *=(a: Double): DVector = {
      BLAS.scal(a, x)
      x
    }

    def +(y: DVector): DVector = {
      val z = y.copy
      BLAS.axpy(1.0, x, z)
      z
    }

    def *(a: Double): DVector = {
      val y = x.copy
      BLAS.scal(a, y)
      y
    }

    def max(y: DVector): DVector = {
      var i = 0
      val N = x.size
      val r = Array.ofDim[Double](N)
      while (i < N) {
        r(i) = Math.max(x(i), y(i))
        i += 1
      }
      dense(r)
    }

    def min(y: DVector): DVector = {
      var i = 0
      val N = x.size
      val r = Array.ofDim[Double](N)
      while (i < N) {
        r(i) = Math.min(x(i), y(i))
        i += 1
      }
      dense(r)
    }
  }

}

