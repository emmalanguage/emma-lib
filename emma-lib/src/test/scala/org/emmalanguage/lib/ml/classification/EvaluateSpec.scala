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

import api._
import lib.ml._
import lib.linalg._
import evaluate._

class EvaluateSpec extends lib.BaseLibSpec {
  type Point = (Double, Double)
  val tolerance = 1e-3
  val h: DVector => Boolean = _.values.head >= 0.0

  /*
   *  TP | FP
   *  ---|---
   *  FN | TN
   */
  def randomValue = math.signum(scala.util.Random.nextDouble() - 0.5)

  val testSet: Seq[Point] = for {
    _ <- 1 to 1000
  } yield (randomValue, randomValue)

  protected def testBag(testSet: Seq[Point]) = {
    val xs = testSet.map {
      case (x, y) =>
        LDPoint(
          "id",
          dense(Array(y)),
          x < 0.0
        )
    }
    DataBag(xs)
  }

  it should "compute precision" in {
    val exp = testPrecision(testSet)
    val act = precision(h, testSet)
    act should equal (exp +- tolerance)
  }

  it should "compute recall" in {
    val exp = testRecall(testSet)
    val act = recall(h, testSet)
    act should equal (exp +- tolerance)
  }

  it should "compute f1score" in {
    val expPrecision = testPrecision(testSet)
    val expRecall = testRecall(testSet)
    val exp = 2.0 * (expPrecision * expRecall) / (expPrecision + expRecall)
    val act = f1score(h, testSet)
    act should equal (exp +- tolerance)
  }

  def precision(h: DVector => Boolean, seq: Seq[Point]) =
    evaluate.precision(h)(testBag(seq))

  def recall(h: DVector => Boolean, seq: Seq[Point]) =
    evaluate.recall(h)(testBag(seq))

  def f1score(h: DVector => Boolean, seq: Seq[Point]) =
    evaluate.f1score(h)(testBag(seq))

  private[lib] def eval(p: Point): Int = p match {
    case (x, y) if x < 0.0 && y >= 0.0  => TP
    case (x, y) if x >= 0.0 && y >= 0.0 => FP
    case (x, y) if x < 0.0 && y < 0.0   => FN
    case (x, y) if x >= 0.0 && y < 0.0  => TN
  }

  private[lib] def testPrecision(seq: Seq[Point]) = {
    val expTp = testSet.count(eval(_) == TP).toDouble
    val expFp = testSet.count(eval(_) == FP).toDouble
    expTp / (expTp + expFp)
  }

  private[lib] def testRecall(seq: Seq[Point]) = {
    val expTp = testSet.count(eval(_) == TP).toDouble
    val expFn = testSet.count(eval(_) == FN).toDouble
    expTp / (expTp + expFn)
  }
}
