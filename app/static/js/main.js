// The JS code builds on the JS code from https://github.com/krishnaik06/Deployment-Deep-Learning-Model
$(document).ready(function () {
    //initially hide predict button
    $('#button-predict').hide();

    // Upload image
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.onload = function (e) {
                $('#imagePreview').css('background-image', 'url(' + e.target.result + ')');
                $('.image-section').show();
                $('#button-predict').show();
                $('#prediction-result').hide();
            }
            reader.readAsDataURL(input.files[0]);
        }
    }

    $("#imageUpload").change(function () {
        readURL(this);
    });

    // Predict dog breed
    $('#button-predict').click(function () {
        $('#prediction-result').hide();
        var form_data = new FormData($('#upload-image')[0]);

        $(this).hide();

        $.ajax({
            type: 'POST',
            url: '/prediction',
            data: form_data,
            contentType: false,
            processData: false,
            success: function (data) {
                $('#prediction-result').show();
                $('#prediction-result').text(data);
            },
        });
    });

});
