<html>
  <head>
    <h2>Matrix multiplication</h2>
  </head>
  <body>
    <h3>Device information:</h3>
    <p><b>CPU</b>: Intel(R) Core(TM) i5-6500 CPU @ 3.20GHz</p>
    <p><b>GPU</b>: NVIDIA GeForce GTX 950</p>
    <p>
      <h3>Results:</h3>
      <table border = "0">
        <caption><h4>Average execution time in seconds</h4></caption>
        <tr>
          <th rowspan = "2">Matrix size</th>
          <th colspan = "2">Device</th>
        </tr>
        <tr>
          <td>CPU</td>
          <td>GPU</td>
        </tr>
        <tr>
          <td>256×256</td>
          <td>0.062591</td>
          <td>0.000441</td>
        </tr>
        <tr>
          <td>512×512</td>
          <td>0.613808</td>
          <td>0.003222</td>
        </tr>
        <tr>
          <td>1024×1024</td>
          <td>9.313177</td>
          <td>0.025037</td>
        </tr>
        <tr>
          <td>2048×2048</td>
          <td>130.522012</td>
          <td>0.178200</td>
        </tr>
        <tr>
          <td>4096×4096</td>
          <td>-</td>
          <td>1.460304</td>
        </tr>
        <tr>
          <td>8192×8192</td>
          <td>-</td>
          <td>13.088716</td>
        </tr> 
      </table>
    </p>
    <p><img src="{{ url_for('video_feed') }}"></p>
  </body>
</html>
