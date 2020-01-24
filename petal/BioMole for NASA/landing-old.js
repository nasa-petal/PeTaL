var baseSVG, baseSelectGroup, basePaintGroup;
// an array of objects containing data about each lifeform, taken from the json file
var lifeforms;

var svgWidth = 330, svgHeight = 330;

// maps that contain all the data about our functions
// each element in the map is keyed by its name
// each result is a map itself with these values:
// - name: the name of the item (duplicates the key for easy querying if you only have the object)
// - parent: a link to its immediate parent group or subgroup (or to 'root' if a group)
// - children: an array of links to its children subgroups or functions (or empty array if a function)
// - matches: an array of indices (into the lifeforms array) for lifeforms that contain this item
var groups = {}, subgroups = {}, functions = {};
var allFunctionItems = {};
var root;

// the max # of lifeforms any group, subgroup, or function has
var maxLifeforms = 0;

var gaussianBlur;
var filter;
var uidiv;
var buttons;
var tip;

var matrixRows = [];
var matrixRowHeight = 150;

var matrixViewOffset = [420,0];

var colorMapper = d3.scaleSequential(d3.interpolateMagma).domain([-3,5]); // purple to orange scheme, the original
//var globalColorArray = ['#b73779','#e75263','#fc8961','#DB7A16']
// var colorMapper = function(v) {
	// return globalColorArray[v];
// }
		
// global mouse variables
var toolBeingDragged = null;
var itemClicked = null;

// lifeform list div
var listHeaderDiv, listDiv;
		

function init() {
	d3.json("json/strategies021218.json", function(error, data) {
		if(error) console.log("error loading json:",error);
		lifeforms = data;
		
		parseData();
							
		// full_post_example.json how many lifeforms are in each group
		//for(let key in groups)
		//	console.log(groups[key].name, groups[key].matches.length);		
	
		// figure out the max # of lifeforms any group, subgroup, or function has
		// we only need to check groups since everything lower cannot have more than its parent
		for(let key in groups)
			if(groups[key].matches.length > maxLifeforms) maxLifeforms = groups[key].matches.length;
		
		uidiv = d3.select('body').append('div')
			.style('margin','1em');
		
		baseSVG = d3.select('body').append('div').style('width','100%')
			.append("svg")
			.attr("id","mysvg")
			//.attr("width",900)
			.attr('width','100%')
			.attr("height",svgHeight)
			.attr("viewBox", "0 0 " + svgWidth*2 + ' ' + svgHeight*2)
			//.on('mousedown',globalMouseDown)
			//.on('mousemove',globalMouseMove)
			.on('mouseup',globalMouseUp);
		
		var defs = baseSVG.append("defs");
		filter = defs.append("filter").attr("id","gooeyCodeFilter");
		gaussianBlur = filter.append("feGaussianBlur")
			.attr("in","SourceGraphic")
			.attr("stdDeviation","8")
			//to fix safari: http://stackoverflow.com/questions/24295043/svg-gaussian-blur-in-safari-unexpectedly-lightens-image
			.attr("color-interpolation-filters","sRGB")
			.attr("result","blur");
		filter.append("feColorMatrix")
			.attr("in","blur")
			.attr("mode","matrix")
			.attr("values","1 0 0 0 0  0 1 0 0 0  0 0 1 0 0  0 0 0 19 -9")
			.attr("result","gooey");
		filter.append("feBlend")
			.attr("in","SourceGraphic")
			.attr("in2","gooey");
		
		var pattern = defs.append('pattern')
				.attr('id','diagonalHatch')
				.attr('width',20)

				.attr('height',20)
				.attr('patternTransform','rotate(45 0 0)')
				.attr('patternUnits','userSpaceOnUse');
			pattern
				.append('line')
				.attr('x1',0)
				.attr('y1',0)
				.attr('x2',0)
				.attr('y2',20)
				.style('stroke','#aaa')
				.style('stroke-width',3);
		
		baseSelectGroup = baseSVG.append('g').attr('id', 'selectGroup').style("filter", "url(#gooeyCodeFilter)");
		
		tip = d3.tip()
		  .attr('class', 'd3-tip')
		  //.offset([-10, 0])
		  //.offset(function(d) { return [d.r,0] })
		  .html(function(d) {
			  var name = d.data.name;
			  if(name === 'ROOT') name = "All Lifeforms"
			  var a = ['<span class="active">' + name + '</span> [' + d.data.matches.length + ']'];
			  var temp = d;
			  while(temp.parent != null && temp.parent.data.name != 'ROOT') {
				  temp = temp.parent;
				  a.push(temp.data.name);
			  }
			  a.reverse();
			  return a.join('<br>');
			  //return "blah";
		  });
		
		baseSVG.call(tip);
		
		createUI();

        updateCircles(true);

        d3.selectAll("circle").each(function(d)  {
            d.data.expandedCircle = true;
            // d.data.children.forEach(e => e.expandedCircle = true);
        });

        updateCircles();

		// let b = d3.select('body');
		// var landingText = b.append(b.select('div'));
        $('.landingText').insertAfter('.d3-tip');
		  // .classed('landingText',true);

		// landingText.html("hello there, this is some landing text. Please enjoy the <a href='index-old.html'>Ask Nature tool</a>");
		logEvent("Opened landing page");
	})
}


function wrapColor(text, color) {
	return '<span style="color:' + color + '">' + text + '</span>';
}

function createUI() {
	buttons = [
		{name: "Gooey", value: true},
		{name: "Movement", value: true},
		{name: "Gaussian", value: 6},
		{name: "feBlend", value: true},
		{name: "Blur Effect", value: "19 -9"}
	];
	
}

function getSetting(name) {
	var match = buttons.filter(d => d.name == name)
	return match.length == 1 ? match[0].value : null;
}

// parse the "Function" item in the json to create a branching tree of interlinked groups, subgroups, functions
// also, change the "Function" item from a string into an array of our "function" objects
function parseData() {
	var maps = [groups,subgroups,functions];
	root = {"name": "ROOT", "parent": null, "children": [], "matches": [], "id": 0, 
		"expandedCircle": false, "expandedMatrix": true}
	
	var id = 1;
	lifeforms.forEach((d,lifeformIndex) => {
		root.matches.push(lifeformIndex);
		var newfuncs = []; // we'll change the function string to an array of objects
		var funcs = d.Functions.split('|');
		funcs.forEach(f => {
			var split = f.split('>');
			// ignore all entries that don't have group > subgroup > function format
			if(split.length == 3) {
				var items = [];
				// pull the items from the groups, or create them if they don't exist
				for(var i=0; i<split.length; i++) {
					var map = maps[i];
					if(!map.hasOwnProperty(split[i])) 
						map[split[i]] = {
							"name": split[i],
							"parent": null,
							"children": [],
							"matches": [],
							"id": id++,
							"expandedCircle": false,
							"expandedMatrix": false
						};
					items.push(map[split[i]]);
				}
				// items now contains 3 items [group, subgroup, function], let's tie them together for this lifeform
				items.forEach((item,i) => {
					if(i>0) 
						item.parent = items[i-1];
					if(i<2 && !item.children.includes(items[i+1])) 
						item.children.push(items[i+1]);
					if(!item.matches.includes(lifeformIndex))
						item.matches.push(lifeformIndex);
				});
				newfuncs.push(items[2]); // save the function item to put back in this lifeform
			}
		})
		
		d.Functions = newfuncs; // change the "Functions" item from a string to an array of our 'function' objects
	})
	
	for(var key in groups) { 
		root.children.push(groups[key])
		groups[key].parent = root;	
	}
	
	// add everything to 'allFunctionItems'
	// do we really need to split it into groups, subgroups, functions?
	[groups,subgroups,functions].forEach(f => {
		for(var key in f) allFunctionItems[key] = f[key];
	})
	
}

function globalMouseDown() {
	// prevents a bunch of weird highlighting of text/elements which breaks our mouse interaction
	d3.event.preventDefault();
}

function globalMouseMove() {
	if(toolBeingDragged != null) {
		var m = d3.mouse(this);
		toolBeingDragged.setPosition(m)
		
	}
}

function globalMouseUp() {
	if(toolBeingDragged != null) {
		var m = d3.mouse(this);
		toolBeingDragged.stopDragging();
		toolBeingDragged = null;
	}
}

function logEvent(str) {
	console.log("Event:",str);
    let url = '/logging/' + str;
    $.post(url, {}, function() {});
}

