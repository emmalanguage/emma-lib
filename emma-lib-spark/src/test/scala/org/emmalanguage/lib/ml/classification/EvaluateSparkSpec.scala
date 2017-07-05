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
package lib.ml.classification

import api.Meta.Projections._
import api._
import lib.linalg._

class EvaluateSparkSpec extends EvaluateSpec with SparkAware {
  override def precision(h: DVector => Boolean, seq: Seq[Point]) = {
    withDefaultSparkSession(implicit spark => emma.onSpark {
      evaluate.precision(h)(testBag(seq))
    })
  }

  override def recall(h: DVector => Boolean, seq: Seq[Point]) = {
    withDefaultSparkSession(implicit spark => emma.onSpark {
      evaluate.recall(h)(testBag(seq))
    })
  }

  override def f1score(h: DVector => Boolean, seq: Seq[Point]) = {
    withDefaultSparkSession(implicit spark => emma.onSpark {
      evaluate.f1score(h)(testBag(seq))
    })
  }
}
