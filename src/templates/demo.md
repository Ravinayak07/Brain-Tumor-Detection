this is the code. How to reset the result when user uploads a new image

<!DOCTYPE html>
<html>

<head>
  <meta charset="UTF-8">
  <title>Binary Classification</title>
  <script src="https://code.jquery.com/jquery-3.5.1.min.js"></script>
  <script>
    function display_result(result) {
      var res = document.getElementById('result');
      res.innerHTML = result.result;
      if (result.result == "Tumor detected") {
        res.style.color = "red";
      }
      else res.style.color = "green";
    }

    $(function () {
      // Function to preview the uploaded image
      function preview_image(event) {
        var reader = new FileReader();
        reader.onload = function () {
          var output = document.getElementById('image-preview');
          output.src = reader.result;
        }
        reader.readAsDataURL(event.target.files[0]);
      }
      // Bind the function to the file input
      var file_input = document.getElementById('image-input');
      file_input.addEventListener('change', preview_image);

      // Bind the function to the form submit event
      $('form').on('submit', function (e) {
        e.preventDefault();
        var form_data = new FormData(this);
        $.ajax({
          type: 'POST',
          url: '/binary.html',
          data: form_data,
          cache: false,
          contentType: false,
          processData: false,
          success: function (data) {
            display_result(data);
          }
        });
      });
    });

  </script>
</head>

<body>
  <nav>
    <h1>Binary Classification</h1>
  </nav>
  <form method="post" enctype="multipart/form-data">
    <input type="file" name="image" id="image-input" accept="image/*" required>
    <img id="image-preview" width="300" height="300">
    <button type="submit">Detect</button>
    <h2 id="result"></h2>
  </form>
</body>

</html>
