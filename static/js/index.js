window.HELP_IMPROVE_VIDEOJS = false;

document.addEventListener('DOMContentLoaded', function () {
    var options = {
        slidesToScroll: 1,
        slidesToShow: 1,
        loop: true,
        infinite: true,
        autoplay: false,
        autoplaySpeed: 5000,
    };

    // Initialize all div elements with the carousel class
    var carousels = bulmaCarousel.attach('.carousel', options);
});
