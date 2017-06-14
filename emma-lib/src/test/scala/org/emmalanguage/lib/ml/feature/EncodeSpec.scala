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

import collection.Map

class EncodeSpec extends FeatureSpec {

  def assertEqual(lhs: Seq[SPoint[Int]], rhs: Array[SPoint[Int]]): Unit = {
    lhs.map(_.id) shouldEqual rhs.map(_.id)
    for ((v, w) <- lhs.map(_.pos) zip rhs.map(_.pos)) {
      v.dat.toList shouldEqual w.dat.toList
      v.idx.toList shouldEqual w.idx.toList
      v.size shouldEqual w.size
    }
  }

  it should "encode frequencies with a hash function" in {
    val exp = exp1(false)
    val act = freq(tokenss.zipWithIndex)
    assertEqual(act, exp)
  }

  it should "encode frequency with a dictionary" in {
    val dic = dict(tokenss)
    val exp = exp2(dic, false)
    val act = freq(dic, tokenss.zipWithIndex)
    assertEqual(act, exp)
  }

  it should "encode binary with a hash function" in {
    val exp = exp1(true)
    val act = bin(tokenss.zipWithIndex)
    assertEqual(act, exp)
  }

  it should "encode binary with a dictionary" in {
    val dic = dict(tokenss)
    val exp = exp2(dic, true)
    val act = bin(dic, tokenss.zipWithIndex)
    assertEqual(act, exp)
  }

  protected final def exp1(bin: Boolean) = for {
    (tokens, id) <- tokenss.zipWithIndex
  } yield {
    val kx = (x: String) => util.nonNegativeMod(encode.native(x), encode.card)
    val rs = tokens.groupBy(kx).mapValues(_.length)
    val vb = new VectorBuilder[Double](encode.card)
    rs.foreach({ case (k, v) => vb.add(k, if (bin) 1.0 else v) })
    SPoint(id, vb.toSparseVector())
  }

  protected final def exp2(dict: Map[String, Int], bin: Boolean) = for {
      (tokens, id) <- tokenss.zipWithIndex
    } yield {
      val kx = (x: String) => util.nonNegativeMod(dict(x), dict.size)
      val rs = tokens.groupBy(kx).mapValues(_.length)
      val vb = new VectorBuilder[Double](dict.size)
      rs.foreach({ case (k, v) => vb.add(k, if (bin) 1.0 else v) })
      SPoint(id, vb.toSparseVector())
    }

  protected def dict(xs: Seq[Array[String]]): Map[String, Int] = {
    encode.dict(for {
      x <- DataBag(xs)
      f <- DataBag(x)
    } yield f)
  }

  protected def freq(xs: Seq[(Array[String], Int)]) = {
    val rs = for {
      (tokens, id) <- DataBag(xs)
    } yield SPoint(id, encode.freq[String]()(tokens))
    rs.collect()
  }

  protected def freq(dict: Map[String, Int], xs: Seq[(Array[String], Int)]) = {
    val rs = for {
      (tokens, id) <- DataBag(xs)
    } yield SPoint(id, encode.freq[String](dict)(tokens))
    rs.collect()
  }

  protected def bin(xs: Seq[(Array[String], Int)]) = {
    val rs = for {
      (tokens, id) <- DataBag(xs)
    } yield SPoint(id, encode.bin[String]()(tokens))
    rs.collect()
  }

  protected def bin(dict: Map[String, Int], xs: Seq[(Array[String], Int)]) = {
    val rs = for {
      (tokens, id) <- DataBag(xs)
    } yield SPoint(id, encode.bin[String](dict)(tokens))
    rs.collect()
  }
}
