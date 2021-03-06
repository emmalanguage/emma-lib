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
package lib.stats

import api._
import lib.linalg._

class StatsSparkSpec extends StatsSpec with SparkAware {

  override protected def basic(xs: Seq[DVector]) =
    withDefaultSparkSession(implicit spark => emma.onSpark {
      summarize(
        stat.count,
        stat.sum(nDim),
        stat.min(nDim),
        stat.max(nDim)
      )(DataBag(xs))
    })

  override protected def moments(xs: Seq[DVector]) =
    withDefaultSparkSession(implicit spark => emma.onSpark {
      summarize(
        stat.mean(nDim),
        stat.variance(nDim),
        stat.stddev(nDim)
      )(DataBag(xs))
    })
}
