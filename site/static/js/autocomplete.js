
$.ajaxSetup({
    beforeSend: function (xhr, settings) {
        function getCookie(name) {
            var cookieValue = null;
            if (document.cookie && document.cookie != '') {
                var cookies = document.cookie.split(';');
                for (var i = 0; i < cookies.length; i++) {
                    var cookie = jQuery.trim(cookies[i]);
                    // Does this cookie string begin with the name we want?
                    if (cookie.substring(0, name.length + 1) == (name + '=')) {
                        cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                        break;
                    }
                }
            }
            return cookieValue;
        }
        if (!(settings.url) || settings.url) {
            // Only send the token to relative URLs i.e. locally.
            xhr.setRequestHeader("X-CSRFToken", getCookie('csrftoken'));
        }
    }
});

// Ajax for posting query
function create_post(query) {
    return $.ajax({
        url: "/bird/api/", // the endpoint
        type: "POST", // http method
        data: { user_input: query }, // data sent with the post request
    })
};

function autocomplete(inp) {
    /*this autocomplete function takes the text field element*/

    var currentFocus;

    /*execute a function when someone writes in the text field:*/
    inp.addEventListener("input", function (e) {
        var a, b, i, val = this.value;

        /*search neo4j with input value*/
        $.when(create_post(this.value)).done(function (search_result) {
            /*close any already open lists of autocompleted values*/
            closeAllLists();
            // if (!val) { return false; }
            currentFocus = -1;

            /*check search resulted in any results:*/
            if (search_result["articles"].length > 0) {
                var arr = []
                for (i = 0; i < search_result["articles"].length; i++) {
                    arr.push(search_result["articles"][i]["title"])
                }
                /*create a DIV element that will contain the items (values):*/
                a = document.createElement("DIV");
                a.setAttribute("id", this.id + "autocomplete-list");
                a.setAttribute("class", "autocomplete-items");
                /*append the DIV element as a child of the autocomplete container:*/
                inp.parentNode.appendChild(a);

                /*for each item in the array...*/
                for (i = 0; i < arr.length; i++) {
                    /*create a DIV element for each matching element:*/
                    b = document.createElement("DIV");
                    /*makes the starting letters bold:*/
                    // b.innerHTML = "<strong>" + arr[i].substr(0, val.length) + "</strong>";
                    // b.innerHTML += arr[i].substr(val.length);
                    b.innerHTML = arr[i];
                    /*insert a input field that will hold the current array item's value:*/
                    b.innerHTML += "<input type='hidden' value='" + arr[i] + "'>";
                    /*execute a function when someone clicks on the item value (DIV element):*/
                    b.addEventListener("click", function (e) {
                        /*insert the value for the autocomplete text field:*/
                        inp.value = this.getElementsByTagName("input")[0].value;
                        /*close the list of autocompleted values,
                        (or any other open lists of autocompleted values:*/
                        closeAllLists();
                    });
                    a.appendChild(b);
                }
            }
        });

    });

    // /*execute a function presses a key on the keyboard:*/
    // inp.addEventListener("keydown", function (e) {
    //     var x = document.getElementById(this.id + "autocomplete-list");
    //     if (x) x = x.getElementsByTagName("div");
    //     if (e.keyCode == 40) {
    //         /*If the arrow DOWN key is pressed,
    //         increase the currentFocus variable:*/
    //         console.log("down was pressed")
    //         currentFocus++;
    //         console.log(currentFocus)
    //         /*and and make the current item more visible:*/
    //         addActive(x);
    //     } else if (e.keyCode == 38) { //up
    //         /*If the arrow UP key is pressed,
    //         decrease the currentFocus variable:*/
    //         currentFocus--;
    //         /*and and make the current item more visible:*/
    //         addActive(x);
    //     } else if (e.keyCode == 13) {
    //         /*If the ENTER key is pressed, prevent the form from being submitted,*/
    //         e.preventDefault();
    //         if (currentFocus > -1) {
    //             /*and simulate a click on the "active" item:*/
    //             if (x) x[currentFocus].click();
    //         }
    //     }
    // });

    // function addActive(x) {
    //     /*a function to classify an item as "active":*/
    //     if (!x) return false;
    //     /*start by removing the "active" class on all items:*/
    //     removeActive(x);
    //     if (currentFocus >= x.length) currentFocus = 0;
    //     if (currentFocus < 0) currentFocus = (x.length - 1);
    //     /*add class "autocomplete-active":*/
    //     x[currentFocus].classList.add("autocomplete-active");
    // }
    // function removeActive(x) {
    //     /*a function to remove the "active" class from all autocomplete items:*/
    //     for (var i = 0; i < x.length; i++) {
    //         x[i].classList.remove("autocomplete-active");
    //     }
    // }
    function closeAllLists(elmnt) {
        /*close all autocomplete lists in the document,
        except the one passed as an argument:*/
        var x = document.getElementsByClassName("autocomplete-items");
        for (var i = 0; i < x.length; i++) {
            if (elmnt != x[i] && elmnt != inp) {
                x[i].parentNode.removeChild(x[i]);
            }
        }
    }

    /*execute a function when someone clicks in the document:*/
    document.addEventListener("click", function (e) {
        closeAllLists(e.target);
    });
}

function create_options(data, query) {

}

/*initiate the autocomplete function on the "myInput" element, and pass along the countries array as possible autocomplete values:*/
autocomplete(document.getElementById("myInput"));
