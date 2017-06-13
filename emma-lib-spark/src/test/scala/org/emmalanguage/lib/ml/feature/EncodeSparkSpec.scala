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
import lib.ml.SPoint

class EncodeSparkSpec extends EncodeSpec with SparkAware {

  override protected def count(xs: Seq[(Array[String], Int)]) =
    withDefaultSparkSession(implicit spark => emma.onSpark {
      val rs = for {
        (tokens, id) <- DataBag(xs)
      } yield SPoint(id, encode.count[String]()(tokens))
      rs.collect()
    })

  override protected def bin(xs: Seq[(Array[String], Int)]) =
    withDefaultSparkSession(implicit spark => emma.onSpark {
      val rs = for {
        (tokens, id) <- DataBag(xs)
      } yield SPoint(id, encode.bin[String]()(tokens))
      rs.collect()
    })
}
