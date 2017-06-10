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
import lib.ml._
import lib.ml.util

import breeze.linalg._

class HashSpec extends FeatureSpec {

  it should "hash count" in {
    val exp = hashes(false)
    val act = count(tokenss.zipWithIndex)
    act.map(_.id) shouldEqual exp.map(_.id)
    for ((v, w) <- act.map(_.pos) zip exp.map(_.pos)) {
      v.dat.toList shouldEqual w.dat.toList
      v.idx.toList shouldEqual w.idx.toList
      v.size shouldEqual w.size
    }
  }

  it should "hash bin" in {
    val exp = hashes(true)
    val act = bin(tokenss.zipWithIndex)
    act.map(_.id) shouldEqual exp.map(_.id)
    for ((v, w) <- act.map(_.pos) zip exp.map(_.pos)) {
      v.dat.toList shouldEqual w.dat.toList
      v.idx.toList shouldEqual w.idx.toList
      v.size shouldEqual w.size
    }
  }

  protected final def hashes(bin: Boolean) = for {
    (tokens, id) <- tokenss.zipWithIndex
  } yield {
    val kx = (x: String) => util.nonNegativeMod(hash.native(x), hash.card)
    val rs = tokens.groupBy(kx).mapValues(_.length)
    val vb = new VectorBuilder[Double](hash.card)
    rs.foreach({ case (k, v) => vb.add(k, if (bin) 1.0 else v) })
    SPoint(id, vb.toSparseVector())
  }

  protected def count(xs: Seq[(Array[String], Int)]) = {
    val rs = for {
      (tokens, id) <- DataBag(xs)
    } yield SPoint(id, hash.count[String]()(tokens))
    rs.collect()
  }

  protected def bin(xs: Seq[(Array[String], Int)]) = {
    val rs = for {
      (tokens, id) <- DataBag(xs)
    } yield SPoint(id, hash.bin[String]()(tokens))
    rs.collect()
  }
}
