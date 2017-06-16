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
package lib.ml.feature

import api._
import lib.ml.FPoint

class TokenizeSpec extends FeatureSpec {

  it should "tokenize gaps" in {
    val exp = for {
      (f, i) <- tokenss.zipWithIndex
    } yield FPoint(i, f.map(_.toLowerCase).toSeq)
    val act = gaps(texts.zipWithIndex)
    act shouldEqual exp
  }

  it should "tokenize tokens" in {
    val exp = for {
      (f, i) <- tokenss.zipWithIndex
    } yield FPoint(i, f.map(_.toLowerCase).toSeq)
    val act = words(texts.zipWithIndex)
    act shouldEqual exp
  }

  protected def gaps(xs: Seq[(String, Int)]) = {
    val rs = for {
      (text, id) <- DataBag(xs)
    } yield FPoint(id, tokenize.gaps()(text).toSeq)
    rs.collect()
  }

  protected def words(xs: Seq[(String, Int)]) = {
    val rs = for {
      (text, id) <- DataBag(xs)
    } yield FPoint(id, tokenize.words()(text).toSeq)
    rs.collect()
  }
}
