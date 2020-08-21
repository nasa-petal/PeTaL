

window.onload = loadData();
  
var tertiaryE2BData;
  
function loadData() {
  loadE2BData();
  loadPrimaryDropDown();
}
  
function getPrimaryDropdownData() {
  return bird_data;
}
  
function getSecondaryDropdownData(primaryIndex) {
  return bird_data[primaryIndex].secondary;
}
  
function getTertiaryDropdownData(primaryIndex, secondaryIndex) {
  return bird_data[primaryIndex].secondary[secondaryIndex].tertiary;
}
  
function loadPrimaryDropDown() {
  var primaryInfo = getPrimaryDropdownData();
  
  for (var i = 0; i < primaryInfo.length; i++) {
    $("#primary-dropdown").append($('<option>', {
      value: primaryInfo[i].id,
      text: primaryInfo[i].name
    }));
  }
}
  
/**
 * Loads Engineering to Biology Thesaurus
*/
function loadE2BData() {
  $.getJSON("../../static/js/bird-bio-words.json", function(data) {
    tertiaryE2BData = data.tertiary;
  });
}
/**
 * Retrieves biological terms for selected tertiary term
 * @param {string} tertiaryId
 * @return {string} result String of all biology terms in format term1, term2, term3
*/
function getBioTerms(tertiaryName) {
    var result = "";
  for (var i = 0; i < tertiaryE2BData.length; i++) {
    if (tertiaryE2BData[i].engineer === tertiaryName)
    var biologistTerms = tertiaryE2BData[i].biologist
  }
  biologistTerms.forEach(function (element) {
    result += element + ", ";
  });
  
  result = result.substring(0, result.length - 2);
  
  return result;
}
  
/**
 * Populates secondary-dropdown based on primary-dropdown selection.
 * Enables secondary-dropdown, disables tertiary-dropdown and search-btn, hides bio-terms.
*/
function primarySelected(primaryIndex) {
  var secondarydd = $("#secondary-dropdown");
  var tertiarydd = $("#tertiary-dropdown");
  var bioTermsDiv = $("#bio-terms");
  var searchBtn = $('#search-btn');
  
  bioTermsDiv.addClass('vis-hidden');
  
  secondarydd.selectedIndex = 0;
  
  $('#secondary-dropdown option[value!=""]').remove();  //remove all but the default option
  $('#tertiary-dropdown option[value!=""]').remove();
  
  if (primaryIndex === "") {
    secondarydd.prop('disabled', true);
    tertiarydd.prop('disabled', true);
  }
  else {
    var secondaryInfo = getSecondaryDropdownData(primaryIndex)
  
    for (var i = 0; i < secondaryInfo.length; i++) {
      secondarydd.append($('<option>', {
        value: secondaryInfo[i].id,
        text: secondaryInfo[i].name
      }));
    }
  
    secondarydd.prop('disabled', false);
    tertiarydd.prop('disabled', true);
    searchBtn.prop('disabled', true);
  }
}
  
/**
 * Populates tertiary-dropdown based on secondary-dropdown selection.
 * Enables tertiary-dropdown, disables search-btn, hides bio-terms.
*/
function secondarySelected(secondaryIndex) {
  var tertiarydd = $("#tertiary-dropdown");
  var primaryIndex = $("#primary-dropdown option:selected").prop("value");
  var bioTermsDiv = $("#bio-terms");
  var searchBtn = $('#search-btn');
  
  bioTermsDiv.addClass('vis-hidden');
  
  $('#tertiary-dropdown option[value!=""]').remove();
  if (secondaryIndex === "") {
    tertiarydd.prop('disabled', true);
  }
  else {
    let tertiaryInfo = getTertiaryDropdownData(primaryIndex, secondaryIndex)
    for (var i = 0; i < tertiaryInfo.length; i++) {
      //var term = getTertiaryTerm(tertiaryInfo[i].name);
      tertiarydd.append($('<option>', {
        value: tertiaryInfo[i].name.toLowerCase(),
        text: tertiaryInfo[i].name
      }));
    }
  
    tertiarydd.prop('disabled', false);
    searchBtn.prop('disabled', true);
  }
}
  
/**
 * Displays biological terms associated with selected tertiary term.
 * Enables search-btn.
*/
function tertiarySelected(tertiaryIndex) {
  var primaryIndex = $("#primary-dropdown option:selected").prop("value");
  var secondaryIndex = $("#secondary-dropdown option:selected").prop("value");
  var tertiaryName = $("#tertiary-dropdown option:selected").prop("text");
  var bioTermsDiv = $("#bio-terms");
  var searchBtn = $('#search-btn');
  var query = $('#to-bioterms')

  var bioTerms = getBioTerms(tertiaryName);
  var newString = "Biology terms in this search: " + bioTerms;
  

  query.append($('<input>', {
    name: 'q',
    type: 'hidden',
    value: bioTerms.toLowerCase(),
  }));
  bioTermsDiv.text(newString);
  bioTermsDiv.removeClass('vis-hidden');
  searchBtn.prop('disabled', false);
}