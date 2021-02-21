$(document).ready(function () {
    // Upload image
    function readURL(input) {
        if (input.files && input.files[0]) {
            var reader = new FileReader();
            reader.readAsDataURL(input.files[0]);
        }
    }

        $.ajax({
            type: 'POST',
            url:   '/prediction',
            data: form_data,
            contentType: false,
            cache: false,
            processData: false,
            success: function(data){
            },

        })

});
