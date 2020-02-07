// represents the object that you move around to paint with
// can be spawned by grabbing something in the selectView or a cell in the paintView

function PaintTool(clickPos, filterList, lifeformList, name, fillColor) {
	this.lifeforms = lifeformList;
	this.filterList = uniqueArray(filterList);
	this.name = name;
	if(!this.name) this.name = generatePaintToolName(this.filterList);
	this.row = null; // currently has no home
	this.isDragging = true;
	this.fillColor = fillColor;
	
	this.group = this.paintIntoGroup(clickPos, baseSVG);
	
}

function generatePaintToolName(filterList) {
	var f = uniqueArray(filterList);
	// remove "All Lifeforms" from the filter list if we have more than 1 item
	if(f.length > 1) {
		var i = f.indexOf("ROOT");
		if(i >= 0) f.splice(i, 1);
	}
	return f.join(" & ");
}

// this is the same as generatePaintToolName, except we surround each name with a span of color
// the better way is probably to just store the depth of each name when we pass it in, to query the color
// but it is decently fast to just search all the arrays so it's fine
function generateColoredPaintToolName(filterList) {
	var f = uniqueArray(filterList);
	// remove "All Lifeforms" from the filter list if we have more than 1 item
	if(f.length > 1) {
		var i = f.indexOf("ROOT");
		if(i >= 0) f.splice(i, 1);
	}
	
	var s = [];
	var g = [{'ROOT': true}, t.groups, t.subgroups, t.functions];
	
	f.forEach(name => {
		g.forEach((list,i) => {
			if(name in list)
				s.push(wrapColor(name,colorMapper(i)));
		})
	})
	
	return s.join(' & ');
}

// creates a group as a child of 'parentElement' positioned relatively at 'pos'
// and draws a representation of the paintTool into it
PaintTool.prototype.paintIntoGroup = function(pos, parentElement) {
	let radius = 8;
	let group = parentElement.append('g')
		.attr('pointer-events','none')
		.attr('transform','translate(' + pos.join(',') + ')')
		.datum(this);

	let circle = group.append('circle')
		.classed('paintToolCircle',true)
		.attr('cx',0)
		.attr('cy',0)
		.attr('r',radius)
		.attr('fill',this.fillColor);

	let text = group.append('text')
		.classed('paintToolText',true)
		.attr('fill',this.fillColor)
		.text(this.name === 'ROOT' ? 'All Lifeforms' : this.name);

	let box = text.node().getBBox();
	text.attr('x',-radius)//-box.width / 2)
		.attr('y',parseInt(circle.attr('r')) + box.height);
		
	return group;
}



// the function that destroys this paint tool
// gets rid of anything persistent
PaintTool.prototype.dispose = function() {
	this.group.remove();
}

PaintTool.prototype.stopDragging = function() {
	var _this = this;
	if(this.row) {
		// find out where the paintTool is anchored in global SVG coordinates
		// so we can move this.group, which is in global coords, to there
		var m = getRelativeXY(0,0,baseSVG.node(),this.row.borderGroup.node())
		this.group
			.transition()
			.ease(d3.easePolyOut)
			.duration(500)
			.attr('transform','translate(' + [m.x,m.y].join(',') + ')')
			//.on('end',function() { _this.group.style('opacity',0); })
			
		// fades out this group from opacity 1 to 0
		// operates on a different ease function, but at the same time, as the transform transition
		linearFadeOut(this.group, 500);
		
	}
	else {
		this.group
			.transition()
			.style('opacity',0)
			.on('end',function() { _this.dispose(); });
	}
}

function linearFadeOut(node, time) {
	var lock = {};	
	d3.select(lock).transition().ease(d3.easeLinear).duration(time)
		.tween("style:opacity", function() {
			var i = d3.interpolate(parseInt(node.style('opacity')),0);
			return function(t) { node.style('opacity',i(t)); }
		})
	
}

PaintTool.prototype.setPosition = function(absolutePos) {
	this.group.attr('transform','translate(' + absolutePos.join(',') + ')')
}