// -----header-----
// scroll
$(document).ready(function() {
    $(window).scroll(function() {
        var sc = $(window).scrollTop()
        if (sc > 150) {
            $("#main-navbar").addClass("navbar-scroll")
        } else {
            $("#main-navbar").removeClass("navbar-scroll")
        }
    });
});
// search box
const searchIcon = document.querySelector(".search-icon");
const close = document.querySelector(".close");

// show more
$(function() {
    const loadMore = document.querySelector("#loadMore");
    x = 12;
    $('.product-items .single-product-items').slice(0, 12).show();
    $('#loadMore').on('click', function(e) {
        e.preventDefault();
        x = x + 100;
        $('.product-items .single-product-items').slice(0, x).slideDown();
        loadMore.classList.add("hide");

    });
});
