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
package lib.ml.feature

import api._
import lib.ml.util

import breeze.linalg._

import collection.Map

@emma.lib
object encode {

  val card = 1 << 10

  val native = (x: Any) => x.hashCode()

  def freq[A](N: Int = card, h: A => Int = native)(xs: Array[A]): SparseVector[Double] =
    encode(N, h, (i: Int, F: Map[Int, Double]) => F.getOrElse(i, 0.0) + 1.0)(xs)

  def bin[A](N: Int = card, h: A => Int = native)(xs: Array[A]): SparseVector[Double] =
    encode(N, h, (_: Int, _: Map[Int, Double]) => 1.0)(xs)

  def apply[A](
    N: Int = card,
    h: A => Int = native,
    u: (Int, Map[Int, Double]) => Double
  )(xs: Array[A]): SparseVector[Double] = {
    var freqs = Map.empty[Int, Double]

    val L = xs.length
    var i = 0
    while (i < L) {
      val x = xs(i)
      val y = util.nonNegativeMod(h(x), N)
      freqs += y -> u(y, freqs)
      i += 1
    }

    val builder = new VectorBuilder[Double](N)
    for ((k, v) <- freqs) builder.add(k, v)
    val rs = builder.toSparseVector(false, true)
    rs
  }
}
