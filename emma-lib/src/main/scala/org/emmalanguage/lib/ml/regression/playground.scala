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
package lib.ml.regression

import api.DataBag
import lib.ml.LDPoint
import lib.ml.kfold
import lib.linalg._
import lib.ml.optimization._

object playground extends App {
  val a = +1.0 // slope
  val b = +7.0 // bias
  val l = -5.0 // lower bound
  val u = +5.0 // upper bound
  val s = +0.5 // step size

  val instances = DataBag(
    for {
      (x, i) <- (l to u by s).zipWithIndex
    } yield
      LDPoint(i.toLong, dense(Array(x)), a * x + b))

  // hyper-parameters
  val miniBatchSize = 10
  val lrRange = 0.1 to 0.5 by 0.1
  val lambdaRange = 0.1 to 0.5 by 0.1
  val maxIter = 10000
  val convergenceTolerance = 1e-5

  val numInstances = instances.size

  val nFolds = 4 // number of splits
  val runs = instances.size / nFolds // number of runs until we have used every split as test-set

  // split the dataset into partitions
  val fractions = Seq.fill(nFolds)(numInstances.toDouble / nFolds.toDouble)
  val splits = kfold.split(fractions)(instances)()

  // construct parameter-grid
  val grid = for (l <- lrRange; la <- lambdaRange) yield (l, la)

  /* OUTER LOOP: run grid search over parameter-grid */
  val builder = Seq.newBuilder[((Double, Double), Double)]
  for {
    lr <- lrRange;
    lambda <- lambdaRange;
    k <- 0 to fractions.size
  } {

    /* INNER LOOP: run one run of k-fold cros validation */

    // parameters for this run
    val train = kfold.except(k)(splits)
    val test = kfold.select(k)(splits)

    val solv = solver.sgd[Long](
      learningRate = lr,
      iterations = maxIter,
      miniBatchSize = miniBatchSize,
      lambda = lambda
    )(
      error.rmse,
      regularization.l2
    )(dense(Array.fill[Double](2)(0.0)))(_)

    val model = linreg.train(train, solv)
    val loss = linreg.predict(model.weights, error.rmse)(test)

    builder += (lr, lambda) -> loss
  }

  val (bestModel, loss) = builder.result().groupBy(_._1).mapValues(ls => ls.map(_._2).sum / ls.size).minBy(_._2)
  println(bestModel)
}
