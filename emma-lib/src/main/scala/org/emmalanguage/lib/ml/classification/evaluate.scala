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
import lib.ml._

@emma.lib
object evaluate {
  val TP = 0
  val FP = 1
  val FN = 2
  val TN = 3

  type Hypothesis = DVector => Boolean
  type TestBag[ID] = DataBag[LDPoint[ID, Boolean]]

  def apply[ID: Meta](h: Hypothesis)(xs: TestBag[ID]): DataBag[LDPoint[ID, Int]] =
    xs.map { x => x.copy(label = {
      val act = x.label
      val pred = h(x.pos)
      (if (pred) 0 else 2) | (if (act) 0 else 1)
    })}

  def precision[ID: Meta](h: Hypothesis)(xs: TestBag[ID]): Double = {
    val evaluated = evaluate(h)(xs)
    val tp = evaluated.count(_.label == TP).toDouble
    val fp = evaluated.count(_.label == FP).toDouble
    tp / (tp + fp)
  }

  def recall[ID: Meta](h: Hypothesis)(xs: TestBag[ID]): Double = {
    val evaluated = evaluate(h)(xs)
    val tp = evaluated.count(_.label == TP).toDouble
    val fn = evaluated.count(_.label == FN).toDouble
    tp / (tp + fn)
  }

  def f1score[ID: Meta](h: Hypothesis)(xs: TestBag[ID]): Double = {
    val prec = precision(h)(xs)
    val rec = recall(h)(xs)
    2.0 * ((prec * rec) / (prec + rec))
  }
}