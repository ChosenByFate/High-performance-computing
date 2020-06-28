<html>
  <head>
    <h2>Genetic algorithm</h2>
  </head>
  <body>
    <h3>Device information:</h3>
    <p><b>CPU</b>: Intel(R) Core(TM) i5-6500 CPU @ 3.20GHz</p>
    <p><b>GPU</b>: NVIDIA GeForce GTX 950</p>
    <p>
      <h3>Results:</h3>
      <table border = "0">
        <caption><h4>Execution time in seconds. 500 points; 5000 epochs.</h4></caption>
        <tr>
          <th rowspan = "2">Population size</th>
          <th colspan = "2">Device</th>
        </tr>
        <tr>
          <td>CPU</td>
          <td>GPU</td>
        </tr>
        <tr>
          <td>1000</td>
          <td>389.787139</td>
          <td>53.522987</td>
        </tr>
        <tr>
          <td>1500</td>
          <td>583.589690</td>
          <td>57.432390</td>
        </tr>
        <tr>
          <td>2000</td>
          <td>778.694910</td>
          <td>83.149958</td>
        </tr>
      </table>
      <p>Approximation function:</p>
      <p><img src=https://latex.codecogs.com/gif.latex?\dpi{120}\0.00000349689196911344%20*%20x^0%20+%200.00002875808603013037%20*%20x^1%20+%200.00004845544914173874%20*%20x^2%20+%200.00000006302847487254%20*%20x^3%20-0.00000000020273538819%20*%20x^4></p>
      <p>Error = 625.774293</p>
      <h4>Graphic:</h4>
      <p><img src=https://github.com/ChosenByFate/High-performance-computing/blob/master/02%20(Genetic%20algorithm)/Graphics/500%20points.png></p>
      <table border = "0">
        <caption><h4>Execution time in seconds. 1000 points; 2000 epochs.</h4></caption>
        <tr>
          <th rowspan = "2">Population size</th>
          <th colspan = "2">Device</th>
        </tr>
        <tr>
          <td>CPU</td>
          <td>GPU</td>
        </tr>
        <tr>
          <td>1000</td>
          <td>310.074193</td>
          <td>42.444645</td>
        </tr>
        <tr>
          <td>1500</td>
          <td>465.580122</td>
          <td>45.246402</td>
        </tr>
        <tr>
          <td>2000</td>
          <td>657.520273</td>
          <td>65.751609</td>
        </tr>
      </table>
      <p>Approximation function:</p>
      <p><img src=https://latex.codecogs.com/gif.latex?\dpi{120}\12.52614693442646931487%20*%20x^0%20+%201.17959173032063602271%20*%20x^1%20-0.00076556205785166640%20*%20x^2%20-0.00000000000000002603%20*%20x^3%20+%200.00000000000000003479%20*%20x^4></p>
      <p>Error = 478070.3998</p>
      <h3>Graphic:</h3>
      <p><img src=https://github.com/ChosenByFate/High-performance-computing/blob/master/02%20(Genetic%20algorithm)/Graphics/1000%20points.png></p>
    </p>
    <p></p>
  </body>
</html>

