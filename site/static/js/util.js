function add(a,b) {
	return [a[0]+b[0],a[1]+b[1]];
}
function sub(a,b) {
    
	return [a[0]-b[0],a[1]-b[1]];
}
function mult(constant,a) {
	return [constant*a[0],constant*a[1]];
}
function normalize(a) {
	var len = length(a);
	return [a[0]/len,a[1]/len];
}
function length(a) {
	return Math.sqrt(a[0]*a[0]+a[1]*a[1]);
}

function uniqueArray(a) {
	var seen = {};
    return a.filter(function(item) {
        return seen.hasOwnProperty(item) ? false : (seen[item] = true);
    });
}

function uniq(a) {
    return Array.from(new Set(a));
}

function random(min,max) {
	return Math.random() * (max-min) + min;
}

/**
* Get a new XY point in SVG-Space, where X and Y are relative to an existing element.  Useful for drawing lines between elements, for example

* X : the new X with relation to element, 5 would be '5' to the right of element's left boundary.  element.width would be the right edge.
* Y : the new Y coordinate, same principle applies
* svg: the parent SVG DOM element
* element: the SVG element which we are using as a base point.
*/
// taken from: https://stackoverflow.com/questions/26049488/how-to-get-absolute-coordinates-of-object-inside-a-g-group
function getRelativeXY(x, y, svg, element){
  var p = svg.createSVGPoint();
  var ctm = element.getCTM();
  p.x = x;
  p.y = y;
  return p.matrixTransform(ctm);
}