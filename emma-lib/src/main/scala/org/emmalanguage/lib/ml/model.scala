/*
 * Copyright © 2014 TU Berlin (emma@dima.tu-berlin.de)
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
package lib.ml

import api._

import breeze.collection.mutable.SparseArray
import breeze.linalg._

import scala.language.implicitConversions

/** Library-agnostic dense vector representation. */
case class DVector(dat: Array[Double])

object DVector {
  implicit def from(v: DenseVector[Double]): DVector =
    if (v.offset != 0 || v.stride != 1 || v.data.length != v.length) DVector(v.toArray)
    else DVector(v.data) // avoid Arrays.copyOfRange if possible

  implicit def to(v: DVector): DenseVector[Double] =
    DenseVector(v.dat)
}

/** Library-agnostic sparse vector representation. */
case class SVector(idx: Array[Int], dat: Array[Double], size: Int)

object SVector {
  implicit def from(v: SparseVector[Double]): SVector = {
    val idx = java.util.Arrays.copyOf(v.index, v.used) // TODO: consider using v.index directly
    val dat = java.util.Arrays.copyOf(v.data, v.used) // TODO: consider using v.data directly
    SVector(idx, dat, v.array.size)
  }

  implicit def to(v: SVector): SparseVector[Double] =
    new SparseVector(new SparseArray(v.idx, v.dat, v.dat.length, v.size, 0.0))
}

/** Point with identity and a dense vector position. */
case class DPoint[ID](@emma.pk id: ID, pos: DVector)

/** Point with identity and a sparse vector position. */
case class SPoint[ID](@emma.pk id: ID, pos: SVector)

/** Point with identity, a dense vector position, and a label. */
case class LDPoint[ID, L](@emma.pk id: ID, pos: DVector, label: L)

/** Point with identity, a dense vector position, and a label. */
case class LSPoint[ID, L](@emma.pk id: ID, pos: SVector, label: L)

/** Features point. */
case class FPoint[ID, F](@emma.pk id: ID, features: F)
